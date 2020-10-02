import torch.nn.init as init
import pickle
from torch.nn import utils as nn_utils
from util import *
import torch.nn as nn

def sent_split(sent):
    words = []
    sent = nlp(sent)
    for w in sent:
        words.append(w.text.lower())
    return words


def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)


class MLSTM(nn.Module):
    def __init__(self, config):
        super(MLSTM, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(config.embed_dim + config.mask_dim, int(config.l_hidden_size / 2), batch_first=True,
                                num_layers=int(config.l_num_layers / 2), bidirectional=True, dropout=config.l_dropout)
        init_ortho(self.rnn)

    # batch_size * sent_l * dim
    def forward(self, feats, seq_lengths=None):
        '''
        Args:
        feats: batch_size, max_len, emb_dim
        seq_lengths: batch_size
        '''
        pack = nn_utils.rnn.pack_padded_sequence(feats, 
                                                 seq_lengths, batch_first=True)
        lstm_out, _ = self.rnn(pack)
        # Unpack the tensor, get the output for varied-size sentences
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return unpacked


class SimpleCat(nn.Module):
    def __init__(self, config):
        '''
        Concatenate word embeddings and target embeddings
        '''
        super(SimpleCat, self).__init__()
        self.config = config
        self.power = config.power
        self.word_embed = nn.Embedding(config.embed_num, config.embed_dim)
        self.mask_embed = nn.Embedding(2, config.mask_dim)
        self.weight_embed = nn.Embedding(2, config.mask_dim, padding_idx = 0)
        self.dropout = nn.Dropout(config.dropout)

    # input are tensors
    def forward(self, sent, mask):
        '''
        Args:
        sent: tensor, shape(batch_size, max_len, emb_dim)
        mask: tensor, shape(batch_size, max_len)
        '''
        sent = Variable(sent)
        mask = Variable(mask)
        batch_size, max_len = mask.size()
        # Use GloVe embedding
        if self.config.if_gpu:  
            sent, mask = sent.cuda(), mask.cuda()
        # to embeddings
        sent_vec = self.word_embed(sent)  # batch_siz*sent_len * dim
        mask_vec = self.mask_embed(mask)  # batch_size*max_len* dim
        # change mask emb to position emb
        mask_list = mask.tolist()
        len_target = np.sum(mask_list, axis = 1)
        position_ids = []
        position_dict = self.get_position_ids(100)
        for j, sent in enumerate(mask_list):
            position_id = []
            target_id_left = sent.index(1)
            for p in range(max_len):
                if p <= target_id_left:
                    position_id.append(((100 - abs(100-(position_dict[p - target_id_left])))/100)**self.power)
                if target_id_left <= p < target_id_left + len_target[j]:
                    position_id.append(1)
                if target_id_left + len_target[j] < p <= max_len:
                    position_id.append(((100-abs(100 - (position_dict[p - target_id_left - len_target[j]])))/100)**self.power)
            position_ids.append(position_id)
        position_weight = torch.FloatTensor(position_ids).cuda()

        return sent_vec, mask_vec, position_weight

    def get_position_ids(self, max_len):
        position_ids = {}
        position = (max_len - 1) * -1
        position_id = 1
        while position <= max_len - 1:
            position_ids[position] = position_id
            position_id += 1
            position += 1
        return position_ids

    def load_vector(self):
        '''
        Load pre-savedd word embeddings
        '''
        with open(self.config.embed_path, 'rb') as f:
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(self.config.embed_path, vectors.shape))
            self.word_embed.weight.data.copy_(torch.from_numpy(vectors))
            self.word_embed.weight.requires_grad = self.config.if_update_embed
            print('embeddings loaded')

    def reset_binary(self):
        self.mask_embed.weight.data[0].zero_()





