import argparse

from nsml import DATASET_PATH
import nsml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

VOCAB_SIZE = 5004
CLASS_NUM = 10
PAD = 0
MASK = 5001
CLR = 5002
EOS = 5003
NILL_IGNORE = 5005


class MyDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

class essemble_bert(nn.Module) :
    def __init__(self , model1  , model2 ):
        super.__init__()
        self.model1  = model1
        self.model2 = model2


    def forward(self , x, segment_label ):

        x1 = self.model1(x,segment_label)
        x2 = self.model2(x,segment_label)

        return x1+x2

class BertDatatset_classification(Dataset) :

    def __init__(self, x, y , istrain = True , masking = False  ) :
        self.y =y
        self.data = [ ]
        self.segment_label = np.zeros(101)
        self.train = istrain
        self.vocab = VOCAB_SIZE
        self.shop_len  = []
        self.masking = masking
        for idx , x_seg  in enumerate(x) :

            a = np.copy(x_seg)
            # a = np.insert(x_seg, 0, CLR)
            shop_len = np.count_nonzero(a)
            self.shop_len.append(shop_len)
            self.data.append(list(a[:shop_len]))
            if idx < 5 :
                print(a[:shop_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx ):
        # max len  = 100 + 1 (CLR)
        input = np.copy(np.array(self.data[idx]))
        shop_len = np.count_nonzero(input ) + 1
        if self.train :
            input  = np.random.permutation(input)

        input = np.insert(input,0,CLR)

        if self.masking and  shop_len > 5  :
            for i, token in enumerate(input[:shop_len]) :
                prob = random.random()
                if self.masking  and prob < 0.2   and token != CLR :
                    input[i] = MASK

            if shop_len  < 101 :
                input = np.concatenate([input, np.zeros(101 - shop_len)] , axis =  0 )

            return input.astype(np.long) , np.array(self.y[idx]).astype(np.long), self.segment_label.astype(np.long)

        else :

            if shop_len < 101:
                input = np.concatenate([input, np.zeros(101 - shop_len)], axis=0)

            return input.astype(np.long) ,  np.array(self.y[idx]).astype(np.long) ,  self.segment_label.astype(np.long)


class BertDatatset_unlabeled(Dataset) :

    def __init__(self, all_x ,  shuffle = True    ) :

        self.data = [ ]
        self.segment_label = np.zeros(101)
        self.vocab = VOCAB_SIZE
        self.shop_len  = []
        self.shuffle = shuffle
        for idx , x_seg  in enumerate(all_x) :

            a = np.copy(x_seg)
            # a = np.insert(x_seg, 0, CLR)
            shop_len = np.count_nonzero(a)
            self.shop_len.append(shop_len)
            self.data.append(list(a[:shop_len]))
            if idx < 5 :
                print(a[:shop_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx ):
        # max len  = 100 + 1 (CLR)
        input = np.copy(np.array(self.data[idx]))
        shop_len = np.count_nonzero(input ) + 1
        if self.shuffle :
            input_aug  = np.copy(input)
            input_aug  = np.insert(input_aug,0,CLR)
            if shop_len > 5:
                for i, token in enumerate(input_aug[:shop_len-1]):
                    prob = random.random()
                    if prob < 0.2 and token != CLR:
                        input[i] = MASK

        input = np.insert(input,0,CLR)


        if shop_len  < 101 :
                input = np.concatenate([input, np.zeros(101 - shop_len)] , axis =  0 )
                if self.shuffle :
                    input_aug = np.concatenate([input_aug, np.zeros(101 - shop_len)], axis=0)



        if self.shuffle :
            return input.astype(np.long) , input_aug.astype(np.long) , self.segment_label.astype(np.long) , self.segment_label.astype(np.long)
        else :
            return input.astype(np.long), self.segment_label.astype(np.long)
