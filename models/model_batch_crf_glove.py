from BatchLinearChainCRF import LinearChainCrf
import torch.nn.init as init
from torch.nn import utils as nn_utils
from util import *
from Layer import SimpleCat
torch.manual_seed(222)


def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)

class biLSTM(nn.Module):
    def __init__(self, config):
        super(biLSTM, self).__init__()
        self.config = config
        self.rnn = nn.GRU(config.embed_dim + config.mask_dim, int(config.l_hidden_size / 2), batch_first=True,
                          num_layers=int(config.l_num_layers / 2), bidirectional=True)
        init_ortho(self.rnn)

    def forward(self, feats, seq_lengths=None):
        '''
        Args:
        feats: batch_size, max_len, emb_dim
        seq_lengths: batch_size
        '''
        pack = nn_utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True)
        # batch_size*max_len*hidden_dim
        lstm_out, _ = self.rnn(pack)
        # Unpack the tensor, get the output for varied-size sentences
        # padding with zeros
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # batch * sent_l * 2 * hidden_states 
        return unpacked


# consits of three components
class AspectSent(nn.Module):
    def __init__(self, config):
        '''
        LSTM+Aspect
        '''
        super(AspectSent, self).__init__()
        self.config = config
        self.input_dim = config.l_hidden_size  # + config.pos_dim
        kernel_num = config.l_hidden_size  # + config.pos_dim
        reduced_size = int(config.l_hidden_size/4)
        self.conv = nn.Conv1d(self.input_dim, kernel_num, 3, padding=1)  # not used

        self.bilstm = biLSTM(config)
        self.feat2tri = nn.Linear(reduced_size, 2+2)
        self.inter_crf = LinearChainCrf(2+2)
        self.h1linear = nn.Linear(self.input_dim, reduced_size)

        self.feat2tri2 = nn.Linear(reduced_size, 2+2)
        self.inter_crf2 = LinearChainCrf(2+2)
        self.h2linear = nn.Linear(self.input_dim,reduced_size)

        self.feat2tri3 = nn.Linear(reduced_size, 2+2)
        self.inter_crf3 = LinearChainCrf(2+2)
        self.h3linear = nn.Linear(self.input_dim,reduced_size)

        self.feat2tri4 = nn.Linear(reduced_size, 2+2)
        self.inter_crf4 = LinearChainCrf(2+2)
        self.h4linear = nn.Linear(self.input_dim,reduced_size)

        self.feat2label = nn.Linear(kernel_num, 3)
        self.feat2label2 = nn.Linear(self.input_dim*4, 3)

        # gcn - not used for current model
        self.W = nn.ModuleList()
        self.W.append(nn.Linear(config.l_hidden_size, config.l_hidden_size))
        self.W.append(nn.Linear(config.l_hidden_size, config.l_hidden_size))
        self.W.append(nn.Linear(200, 100))

        # cnn - not used for current model
        self.filters = [3]
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1,
                                              out_channels= config.l_hidden_size,
                                              kernel_size= (k, config.l_hidden_size+2),
                                              padding=1) for k in self.filters])
        self.loss = nn.NLLLoss()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout2)
        self.dropout2 = nn.Dropout(config.dropout)
        self.cat_layer = SimpleCat(config)
        self.cat_layer.load_vector()
        self.sigmoid = nn.Sigmoid()

    def get_pos_weight(self, masks, lens):
        '''
        Get positional weight
        '''
        pos_wghts = torch.zeros(masks.size())
        t_num = masks.sum(1)
        for i, m in enumerate(masks):
            begin = m.argmax()
            for j, b in enumerate(m):
                # padding words' weights are zero
                if j > lens[i]:
                    break
                if j < begin:
                    pos_wghts[i][j] = 1 - (begin-j).to(torch.float)/lens[i].to(torch.float)
                if b == 1:
                    pos_wghts[i][j] = 1
                if j > begin + t_num[i]:
                    pos_wghts[i][j] = 1 - (j-begin).to(torch.float)/lens[i].to(torch.float)
        return pos_wghts
    
    def get_target_emb(self, context, masks):
        # Target embeddings
        # Find target indices, a list of indices
        batch_size, max_len, hidden_dim = context.size()
        target_indices, target_max_len = convert_mask_index(masks)
        # Find the target context embeddings, batch_size*max_len*hidden_size
        masks = masks.type_as(context)
        masks = masks.expand(hidden_dim, batch_size, max_len).transpose(0, 1).transpose(1, 2)
        target_emb = masks * context
        target_emb_avg = torch.sum(target_emb, 1)/torch.sum(masks, 1)  # Batch_size*embedding
        return target_emb_avg
    
    def compute_scores(self, sents, masks, lens, is_training=True):
        '''
        Args:
        sents: batch_size*max_len*word_dim
        masks: batch_size*max_len
        lens: batch_size
        '''
        batch_size, max_len = masks.size()
        target_indices, target_max_len = convert_mask_index(masks)
        sents, mask, pos= self.cat_layer(sents, masks)
        sents = self.dropout2(sents)
        sents = torch.cat([sents, mask], 2)
        context = self.bilstm(sents, lens)  # Batch_size*sent_len*hidden_dim
        pos = [x.unsqueeze(1).expand(max_len, self.input_dim) for x in pos]
        pos = torch.stack(pos)
        context = torch.mul(context, pos)

        batch_size, max_len, hidden_dim = context.size()
        word_mask = torch.full((batch_size, max_len), 0)
        for i in range(batch_size):
            word_mask[i, :lens[i]] = 1.0    

        # head 1
        context1 = self.h1linear(context)
        feats1 = self.feat2tri(context1)  # Batch_size*sent_len*2
        marginals1 = self.inter_crf.compute_marginal(feats1, word_mask.type_as(feats1))
        select_polarities1 = [marginal[:, 1] for marginal in marginals1]
        gammas = [sp.sum() for sp in select_polarities1]
        select_polarities1 = [sv/gamma for sv, gamma in zip(select_polarities1, gammas)]        
        sent_vs1 = [torch.mm(sp.unsqueeze(0), context[i, :lens[i], :]) for i, sp in enumerate(select_polarities1)]
        # head 2
        context2 = self.h2linear(context)
        feats2 = self.feat2tri2(context2)  # Batch_size*sent_len*2
        marginals2 = self.inter_crf2.compute_marginal(feats2, word_mask.type_as(feats2))
        select_polarities2 = [marginal[:, 1] for marginal in marginals2]
        gammas = [sp.sum() for sp in select_polarities2]
        select_polarities2 = [sv/gamma for sv, gamma in zip(select_polarities2, gammas)]
        sent_vs2 = [torch.mm(sp.unsqueeze(0), context[i, :lens[i], :]) for i, sp in enumerate(select_polarities2)]

        # head 3
        context3 = self.h3linear(context)
        feats3 = self.feat2tri3(context3)  # Batch_size*sent_len*2
        marginals3 = self.inter_crf3.compute_marginal(feats3, word_mask.type_as(feats3))
        select_polarities3 = [marginal[:, 1] for marginal in marginals3]
        gammas = [sp.sum() for sp in select_polarities3]
        select_polarities3 = [sv/gamma for sv, gamma in zip(select_polarities3, gammas)]
        sent_vs3 = [torch.mm(sp.unsqueeze(0), context[i, :lens[i], :]) for i, sp in enumerate(select_polarities3)]
        # head 4
        context4 = self.h4linear(context)
        feats4 = self.feat2tri4(context4)  # Batch_size*sent_len*2
        marginals4 = self.inter_crf4.compute_marginal(feats4, word_mask.type_as(feats4))
        select_polarities4 = [marginal[:, 1] for marginal in marginals4]
        gammas = [sp.sum() for sp in select_polarities4]
        select_polarities4 = [sv/gamma for sv, gamma in zip(select_polarities4, gammas)] 
        sent_vs4 = [torch.mm(sp.unsqueeze(0), context[i, :lens[i], :]) for i, sp in enumerate(select_polarities4)]

        sent_vs = torch.zeros(batch_size, hidden_dim*4).cuda()
        for i in range(batch_size):
            sent_vs[i] = torch.cat((sent_vs1[i],sent_vs2[i], sent_vs3[i], sent_vs4[i]), dim=1)
        select_polarities = [[marginal[:, 1] for marginal in marginals1], [marginal[:, 1] for marginal in marginals2],
                             [marginal[:, 1] for marginal in marginals3], [marginal[:, 1] for marginal in marginals4]]

        sent_vs = F.relu(self.dropout(sent_vs))
        label_scores = self.feat2label2(sent_vs).squeeze(0)       

        if is_training:
            return label_scores, 0, 0, 0
        else:
            return label_scores, select_polarities, sent_vs, [select_polarities1, select_polarities2,
                                                              select_polarities3, select_polarities4]
    
    def forward(self, sents, masks, labels, lens):
        '''
        inputs are list of list for the convenince of top CRF
        Args:
        sent: a list of sentences， batch_size*len*emb_dim
        mask: a list of mask for each sentence, batch_size*len
        label: a list labels
        '''
        if self.config.if_reset:
            self.cat_layer.reset_binary()
        scores, s_prob, sent_vs, p = self.compute_scores(sents, masks, lens)
        scores = F.log_softmax(scores, 1)  # Batch_size*label_size
        cls_loss = self.loss(scores, labels)
        return cls_loss, 0 

    def predict(self, sents, masks, sent_lens):
        if self.config.if_reset:
            self.cat_layer.reset_binary()
        scores, best_seqs, sent_vs, p = self.compute_scores(sents, masks, sent_lens, False)
        batch, length = sents.size()
        if batch == 1:
            _, pred_label = scores.unsqueeze(0).max(1)
        else:
            _, pred_label = scores.max(1)
        return pred_label, best_seqs, sent_vs, p


def convert_mask_index(masks):
    '''
    Find the indice of none zeros values in masks, namely the target indice
    '''
    target_indice = []
    max_len = 0
    try:
        for mask in masks:
            indice = torch.nonzero(mask == 1).squeeze(1).cpu().numpy()
            if max_len < len(indice):
                max_len = len(indice)
            target_indice.append(indice)
    except:
        print('Mask Data Error')
        print(mask)
    return target_indice, max_len
