import torch
import torch.nn as nn

SOS = "<SOS>"  # start of sequence
EOS = "<EOS>"  # end of sequence
CUDA = torch.cuda.is_available()
torch.manual_seed(222)


class LinearChainCrf(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        # matrix of transition scores from j to i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.SOS_IDX = num_tags-2
        self.EOS_IDX = num_tags-1
        self.transitions.data[self.SOS_IDX, :] = -10000.  # no transition to SOS
        self.transitions.data[:, self.EOS_IDX] = -10000.  # no transition from EOS except to PAD

    def _forward_alg(self, feats, masks):
        '''
        h: batch_size, max_len, tag_size
        mask:batch_size, max_len. binary values
        '''
        assert len(feats) == len(masks)
        batch_size, max_len, _ = feats.size()
        # initialize forward variables in log space
        alpha = []
        alpha_t = Tensor(batch_size, self.num_tags).fill_(-10000.) # [B, C]
        alpha_t[:, self.SOS_IDX] = 0.
        trans = self.transitions.expand(batch_size, 
                                        self.num_tags, self.num_tags) # [b, tag_size, tag_size]
        for t in range(max_len):  # recursion through the sequence
            mask_t = masks[:, t].unsqueeze(1)  # batch_size*1
            emit_t = feats[:, t].unsqueeze(2)  # [batch_size, tag_size, 1]
            score = alpha_t  # [batch_size * tagsize]
            score_t = score.unsqueeze(1) + emit_t + trans  # [B, 1, tag_size] -> [B, tag_size, tag_size]
            score_t = log_sum_exp(score_t)  # [B, C, C] -> [B, C]
            alpha_t = score_t * mask_t + score * (1 - mask_t)  # if mask zero, copy previous alpha
            alpha.append(alpha_t)
        score = log_sum_exp(alpha_t + self.transitions[self.EOS_IDX])
        alpha = torch.stack(alpha, 1)
        return score, alpha  # partition function
    
    def _backward_alg(self, feats, masks): # forward algorithm
        '''
        h: batch_size, max_len, tag_size
        mask:batch_size, max_len. binary values
        '''
        assert len(feats) == len(masks)
        batch_size, max_len, _ = feats.size()
        # initialize forward variables in log space
        beta_t = Tensor(batch_size, self.num_tags).fill_(-10000.) # [B, C]
        beta_t[:, self.EOS_IDX] = 0.
        beta = []
        feats_reversed = torch.zeros_like(feats)
        masks_reversed = torch.zeros_like(masks)
        for i, m in enumerate(masks):
            num = m.sum().long()
            reverse_index = list(reversed(range(num)))
            feats_reversed[i, :num] = feats[i, reverse_index]
            masks_reversed[i, :num] = 1
        # change the start and the end tag
        trans = self.transitions.expand(batch_size, self.num_tags, 
                                  self.num_tags).transpose(1,2)  # [b, tag_size, tag_size]
        # Note reverse the sentence feats,pay attention to the paddings
        for t in range(max_len):  # recursion through the sequence
            mask_t = masks_reversed[:, t].unsqueeze(1)  # batch_size*1
            emit_t = feats_reversed[:, t].unsqueeze(2)  # [batch_size, tag_size, 1]
            score = beta_t  # [batch_size*tagsize]
            score_t = score.unsqueeze(1) + emit_t + trans  # [B, 1, tag_size] -> [B, tag_size, tag_size]
            score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
            beta_t = score_t * mask_t + score * (1 - mask_t)  # if the mask is zero, copy last value
            beta.append(beta_t)  # beta is reversed
        score = log_sum_exp(beta_t + self.transitions[:, self.SOS_IDX])
        beta = torch.stack(beta, 1)
        # reverse beta
        for i, m in enumerate(masks):
            num = m.sum().long()
            reverse_index = list(reversed(range(num)))
            beta[i, :num] = beta[i, reverse_index]
        return score, beta  # partition function
    
    def compute_marginal(self, feats, masks):
        Z1, alpha = self._forward_alg(feats, masks)
        Z2, beta = self._backward_alg(feats, masks)
        num = masks.sum(1).long()
        marginals = []
        for i in range(len(feats)):
            logit = alpha[i, :num[i]] + beta[i, :num[i]] - feats[i, :num[i]]
            denominator = Z1[i]
            marginal = torch.exp(logit-denominator)
            marginals.append(marginal)
        return marginals

    def score(self, feats, tags, masks):  # calculate the score of a given sequence
        batch_size, max_len, _= feats.size()
        score = Tensor(batch_size).fill_(0.)
        feats = feats.unsqueeze(3)
        trans = self.transitions.unsqueeze(2)
        for t in range(max_len):  # recursion through the sequence
            mask_t = masks[:, t]
            emit_t = torch.cat([h[t, y[t + 1]] for h, y in zip(feats, tags)])
            trans_t = torch.cat([trans[y[t + 1], y[t]] for y in tags])
            score += (emit_t + trans_t) * mask_t
        last_tag = tags.gather(1, masks.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.transitions[self.EOS_IDX, last_tag]
        return score

    def decode(self, feats, masks):  # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        assert len(feats) == len(masks)
        batch_size, max_len, _ = feats.size()
        bptr = LongTensor()
        score = Tensor(batch_size, self.num_tags).fill_(-10000.)
        score[:, self.SOS_IDX] = 0.

        for t in range(max_len):  # recursion through the sequence
            mask_t = masks[:, t].unsqueeze(1)  # batch_size*1
            score_t = score.unsqueeze(1) + self.transitions  # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2)  # best previous scores and tags
            score_t = score_t + feats[:, t]  # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score = score + self.transitions[self.EOS_IDX]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(batch_size):
            x = best_tag[b]  # best tag
            y = int(masks[b].sum().item())
            for bptr_t in reversed(bptr[b][:y]):
                x = bptr_t[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path


def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x


def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x


def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x


def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x


def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))
