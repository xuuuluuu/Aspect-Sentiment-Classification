from collections import namedtuple, defaultdict
import torch
import numpy as np
import pickle
import os

SentInst = namedtuple("SentenceInstance", "id text text_ids text_inds opinions")
OpinionInst = namedtuple("OpinionInstance", "target_text polarity class_ind target_mask target_ids target_tokens")


class data_reader:
    def __init__(self, config, is_training=True):
        '''
        Load dataset and create batches for training and testing
        '''
        self.is_training = is_training
        self.config = config

    def load_data(self, load_path):
        '''
        Load the dataset
        '''
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                self.data_batch = pickle.load(f)
                self.data_len = len(self.data_batch)
            self.load_local_dict()
        else:
            print('Data not exist!')  
            return None
        return self.data_batch

    def load_local_dict(self):
        '''
        Load dictionary files
        '''
        if not os.path.exists(self.config.dic_path):
            print('Dictionary file not exist!')
        with open(self.config.dic_path, 'rb') as f:
            word2id, _, _ = pickle.load(f)


class data_generator:
    def __init__(self, config, data_batch, is_training=True):
        '''
        Generate training and testing samples
        Args:
        config: configuration parameters
        data_batch: data list, each contain a nametuple
        '''    
        self.is_training = is_training
        self.config = config
        self.index = 0
        # Filter sentences without targets
        self.data_batch  = self.remove_empty_target(data_batch)
        self.data_len = len(self.data_batch)
        self.UNK = "unk"
        self.EOS = "<eos>"
        self.PAD = "<pad>"
        self.load_local_dict()

    def remove_empty_target(self, data_batch):
        '''
        Remove items without targets
        '''
        original_num = len(data_batch)
        filtered_data = []
        filtered_weights = []
        for i,item in enumerate(data_batch):
            if sum(item[1])>0:
                filtered_data.append(item)
            else:
                print('Mask Without Target', item[0], 'Target', item[5]) 
        return filtered_data

    def load_local_dict(self):
        '''
        Load dictionary files
        '''
        if not os.path.exists(self.config.dic_path):
            print('Dictionary file not exist!')
        with open(self.config.dic_path, 'rb') as f:
            word2id, _, _ = pickle.load(f)
        self.UNK_ID = word2id[self.UNK]
        self.PAD_ID = word2id[self.PAD]
        self.EOS_ID = word2id[self.EOS]

    def reset_samples(self):
        self.index = 0

    def pad_data(self, sents, masks, labels, texts, targets, target_ids, test):
        '''
        Padding sentences to same size
        '''
        sent_lens = [len(tokens) for tokens in sents]
        len_sent = len(sent_lens)
        sent_lens = torch.LongTensor(sent_lens)
        label_list = torch.LongTensor(labels)
        max_len = max(sent_lens)
        batch_size = len(sent_lens)
        # Padding mask
        mask_vecs = np.zeros([batch_size, max_len])
        mask_vecs = torch.LongTensor(mask_vecs)
        for i, mask in enumerate(masks):
            mask_vecs[i, :len(mask)] = torch.LongTensor(mask)
        # padding sent with PAD IDs
        sent_vecs = np.ones([batch_size, max_len]) * self.PAD_ID
        sent_vecs = torch.LongTensor(sent_vecs)
        for i, s in enumerate(sents):  # batch_size*max_len
            sent_vecs[i, :len(s)] = torch.LongTensor(s)
        sent_lens, perm_idx = sent_lens.sort(0, descending=True)
        if test:
            with open('test_id.txt', 'a')as f:
                for i in perm_idx:
                    f.write(str(i.cpu()) + '\n')
                f.write('\n')
                f.close()
        # perm_idx = torch.LongTensor(list(range(len_sent)))
        sent_ids = sent_vecs[perm_idx]
        mask_vecs = mask_vecs[perm_idx]
        label_list = label_list[perm_idx]
        texts = [texts[i.item()] for i in perm_idx]
        targets = [targets[i.item()] for i in perm_idx]
        target_ids = [target_ids[i.item()] for i in perm_idx]
        
        return sent_ids, mask_vecs, label_list, sent_lens, texts, targets, target_ids 

    def get_ids_samples(self, test = False, is_balanced=False):
        '''
        Get samples including ids of words, labels
        '''
        # First get batches of testing data
        if self.data_len - self.index >= self.config.batch_size:
            # print('Sample Index:', self.index)
            start = self.index
            end = start + self.config.batch_size
            samples = self.data_batch[start: end]
            self.index = end
            tokens, mask_list, label_list, token_ids, texts, targets, target_ids = zip(*samples)
            
            # Sorting happens here
            sent_ids, mask_vecs, label_list, sent_lens, texts, targets, target_ids = self.pad_data(token_ids,
                                                                                                   mask_list,
                                                                                                   label_list, texts, 
                                                                                                   targets, target_ids,
                                                                                                   test)
        else:  # Then generate testing data one by one
            samples = self.data_batch[self.index:]
            if self.index == self.data_len - 1:  # if only one sample left
                samples = list(samples)
            tokens, mask_list, label_list, token_ids, texts, targets, target_ids = zip(*samples)
            sent_ids, mask_vecs, label_list, sent_lens, texts, targets, target_ids = self.pad_data(token_ids, 
                                                                                                   mask_list, 
                                                                                                   label_list, 
                                                                                                   texts, 
                                                                                                   targets, 
                                                                                                   target_ids,
                                                                                                   test)
            self.index += len(samples)
        yield sent_ids, mask_vecs, label_list, sent_lens, texts, targets, target_ids

