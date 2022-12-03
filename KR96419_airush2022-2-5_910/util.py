import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import MyDataset
from torch.nn.modules.loss import _WeightedLoss
VOCAB_SIZE = 5000
CLASS_NUM = 10



def loss_func(out, y):
    return F.cross_entropy(out, y)


def get_attn_pad_mask(seq_q, seq_k):
   batch_size, len_q = seq_q.size()
   batch_size, len_k = seq_k.size()
   # eq(zero) is PAD token
   pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
   return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight , ignore_index= 0
        )
        return self.linear_combination(loss / n, nll)


def data_aug(train_x, train_y) :

    buckets  =  [[ ] for _ in range(10)]

    for x,y in zip(train_x, train_y) :
        buckets[y].append(x)

    for label , bucket in enumerate(buckets)  :
            bucket_len = len(bucket)
            print(bucket_len)
            check = [0 for _ in range(bucket_len)]

            for idx , a in enumerate(bucket[:bucket_len]):

                a = a[:np.count_nonzero(a)]

                for next_idx , next in enumerate(bucket[idx+1 : bucket_len])  :

                    b = next[:np.count_nonzero(next)]
                    new_length = len(a) + len(b)
                    if new_length <= 100 and  (len(a)/len(b))  > 0.7 and (len(a)/len(b))  < 1.3  and check[idx] <3 and check[idx] < 3 :
                        check[idx] +=1
                        check[next_idx] += 1
                        a_index   = np.sort(np.random.choice(np.arange(new_length), size=len(a), replace=False))
                        b_index = np.sort(np.delete(np.arange(new_length) , a_index))

                        new = np.array([ 0 for _ in range(new_length)])
                        new[a_index] = a
                        new[b_index] = b
                        real_new = np.array([0 for _ in range(100)])

                        real_new[:new_length] = new
                        buckets[label].append(real_new)

                        # print(a,b, real_new )
                    else :
                        continue

    x_auged = []
    y_auged = []
    for idx , bucket in enumerate(buckets) :
        for x in bucket:
            x_auged.append(list(x))
            y_auged.append(idx)
    return np.array(x_auged), np.array(y_auged)




