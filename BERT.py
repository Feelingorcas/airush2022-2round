import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import tqdm
from tqdm import tqdm
import torch
import random
import os
from sklearn.model_selection import train_test_split
import Bert4rec
from BERT_pytorch.bert_pytorch.model.bert import BERT
from util import LabelSmoothingLoss
from BERT_pytorch.bert_pytorch.model.utils import GELU
import argparse

from nsml import DATASET_PATH
import nsml

VOCAB_SIZE = 5003
CLASS_NUM = 10
PAD = 0
MASK = 5001
CLR = 5002
EOS = 5003
## 5번 부터 이제 원래있던 단어들이 사용되기 시작함
# 1898
# 3102

class BERTessemble(nn.Module) :
    def __init__(self , model1 , model2 ):
        super().__init__()
        self.model1 = model1
        self.model2 = model2


    def forward(self ,x ,segment_label):

        return torch.softmax( self.model1(x,segment_label),dim = 1 ) + torch.softmax( self.model2(x,segment_label) ,  dim = 1 )


def get_vocab_dic(all_x) :
     num_pop  = np.array([0 for _ in range(5001)])
     len1 = 0
     for x_seg in all_x :
         len1 +=    np.count_nonzero(x_seg)
         for x in x_seg :
                 num_pop[x] += 1
     mean_len = len1 / len(all_x)

     vocab_dic   = {}
     dic_num = 0
     for i in num_pop :
         if i >= 1000 :
             dic_num += 1
     dic_num += 3
     assigned = 0

     for idx, num in enumerate(num_pop) :
         if num >= 1000  :
            vocab_dic[idx] = assigned
            assigned += 1
         else :
             vocab_dic[idx] = dic_num
     print(dic_num+1 == VOCAB_SIZE)
     print(dic_num)
     print(VOCAB_SIZE)
     print(vocab_dic)
     # print(vocab_dic)

     return vocab_dic , dic_num+1




def get_BERT(args= None )  :
    VOCAB_SIZE = 5003
    if args.model == 'bert_all' :
         VOCAB_SIZE = 5003
    return BERT(vocab_size= VOCAB_SIZE , hidden= args.hidden_dim , n_layers = args.n_layers , attn_heads = args.attn_heads)

class BERTClassifer(nn.Module) :

    def __init__(self, bert : BERT ,  args , hidden = 768 , freeze = True ):
        super().__init__()
        self.bert = bert
        self.hidden = hidden
        if freeze :
            for n , p in self.bert.named_parameters() :
                    p.requires_grad = False

        self.vocab = VOCAB_SIZE
        # self.layernorm =  nn.LayerNorm()
        self.linear1 = nn.Linear(hidden, args.fc_dim)
        self.linear2 = nn.Sequential(nn.Linear(args.fc_dim , 1-))
        # self.linear3 =  nn.Sequential(nn.Linear(args.fc_dim , 10))
        #
        # self.gelu  = GELU()
        # self.flatten = nn.Flatten(start_dim=1 )
        # self.logsoftmax = nn.LogSoftmax(dim = -1 )
        # self.dropout = nn.Dropout(0.4)

        # self.classifier = nn.Sequential(nn.Linear(hidden, args.fc_dim) , GELU(), nn.Dropout(0.3) ,  nn.Linear(args.fc_dim , args.fc_dim) , GELU() ,nn.Linear(args.fc_dim , 10))

    def forward (self,x , segment_label  )  :

        segment_label2 = (x > 0).long()
        seq_len = segment_label2.sum(dim=1) -1

        x = self.bert(x, segment_label)

        x = x * segment_label2[..., None]

        x = torch.sum(x[:,1:,:], dim=1)

        x = x / seq_len[..., None]

        # x = self.linear1(x)
        #
        # x = self.gelu(x)
        #
        # x = self.dropout(x)
        #
        # x = self.linear2(x)
        #
        # x = self.gelu(x)
        #
        # x = self.linear3(x)

        x = self.classifier(x)

        return x





class BERT_forall(nn.Module) :

    def __init__(self, bert: BERT, shopping_size, all_x=None):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedModel(self.bert.hidden, shopping_size)
        if all_x is not None:
            self.unk_info, self.vocab = get_vocab_dic(all_x)

        self.next_lm = NextModel(self.bert.hidden , shopping_size)

    def forward(self , x, segment_label) :

        x = self.bert(x, segment_label)

        return self.next_lm(x) , self.next_lm(x[:,0,:])


class NextModel(nn.Module)  :

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        # self.gelu = nn.GELU()
        # self.norm = nn.BatchNorm2d()
        # 원래 BERT는 이렇게 되어있음.
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):

        return self.softmax(self.linear(x))
        # return self.linear(x)


class MaskedModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        # self.gelu = nn.GELU()
        # self.norm = nn.BatchNorm2d()
        # 원래 BERT는 이렇게 되어있음.
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):

        return self.softmax(self.linear(x))
        # return self.linear(x)

class BERTMASK(nn.Module):
    """
    BERT Language Model
    Masked Language Model
    """

    def __init__(self, bert: BERT, shopping_size , all_x = None  ):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedModel(self.bert.hidden, shopping_size)
        if all_x is not None  :

            self.unk_info   , self.vocab  = get_vocab_dic(all_x)

    def forward(self, x, segment_label ):
        x = self.bert(x, segment_label) # B*102*384
        return  self.mask_lm(x) # B*102*self.vocah


class BERTdataset(Dataset)  :
    def __init__(self, all_x , shop_len = 100 , mask_factor = 0.2  , origin = False , get_both = False ):
        self.shop_len = shop_len
        self.mask_factor = mask_factor
        self.data = []
        # self.unk_info  = get_pop_number(all_x)
        # self.vocab_dic  , self.vocab = get_vocab_dic(all_x)
        self.last =  []
        self.label = []
        self.origin = origin
        for idx , shopping_list in enumerate(tqdm(all_x)) :
            # shopping_list = np.where(self.unk_info[shopping_list] > 0, shopping_list,  )
            shop_len = np.count_nonzero(shopping_list)
            self.last.append(shopping_list[shop_len-1] if shop_len > 3 else 0 )
            data = shopping_list

            a = np.array(data)
            # a = np.insert(data, 0, CLR)

            # a = np.insert(a, shop_len+1 , EOS)
            self.data.append(list(a))
            # if idx < 5 :
            #
            #     print(a, shop_len , self.last[idx])

        self.data = np.array(self.data)

    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx):

        shopping_len = np.count_nonzero(self.data[idx])
        bert_input , bert_label , mask_idx  = self.random_masking(self.data[idx])
        # segment_label  = np.ones_like(bert_input[:shopping_len])
        # segment_label = np.append(segment_label , np.array([0 for i in range(len(self.data[idx])-shopping_len)]))
        segment_label = np.zeros_like(bert_input)
        bert_input_for_next = np.copy(np.array(self.data[idx]))
        bert_input_for_next[shopping_len-2] = MASK

        # output = {"bert_input": bert_input,
        #           "bert_label": bert_label,
        #           "segment_label": segment_label,
        #             }
        # print(output)
        if self.origin :
            bert_input  = np.copy(self.data[idx])

        # print(type(bert_input), type(bert_label), type(segment_label))
        return bert_input.astype(np.long) , np.array(bert_input_for_next).astype(np.long)  , bert_label.astype(np.long), segment_label.astype(np.long) , self.last[idx]
    def random_masking(self, shopping_log ):
        tokens = np.copy(shopping_log)
        output_label = []
        shop_len = np.count_nonzero(tokens)
        mask_idx = [ ]
        if shop_len < 2 :
            output_label = np.zeros_like(tokens)
            return tokens , output_label , mask_idx

        for i, token in enumerate(tokens[:shop_len]):
            prob = random.random()
            if prob < 0.2 and token != CLR and token != EOS :
                prob /= 0.2
                mask_idx.append(i)

                # 80% randomly change token to mask token
                if prob < 0.8:
                    output_label.append(tokens[i])
                    tokens[i] = MASK

                # 10% randomly change token to random token
                elif prob < 0.9:
                    output_label.append(tokens[i])
                    tokens[i] = random.randint(1,5000)

                # 10% randomly change token to current token
                else:
                    output_label.append(tokens[i])
                    tokens[i] = tokens[i]



            else:
                output_label.append(0)
        for idx in range(len(tokens) - shop_len) :
            output_label.append(0)
        return tokens, np.array(output_label) , np.array(mask_idx)


def train_bert(args, model  : BERTMASK , dataset )  :


    train_dataset, valid_dataset  = train_test_split(dataset, test_size = args.valid_ratio , shuffle= True)

    print(len(train_dataset))
    print(len(valid_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)

    test_dataloader  =torch.utils.data.DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = True )


    print('end making dataloader')

    criterion = torch.nn.CrossEntropyLoss(ignore_index = 0).to(args.device)

    # criterion = LabelSmoothingLoss(smoothing = 0.1 )
    optimizer = torch.optim.Adam(params = model.parameters() , lr = args.lr , weight_decay = 0.0001 )
    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))

    print('train_BERT_mtm_start')
    for epoch in range(args.epoch_max) :
        model.train()
        for idx , data in enumerate(tqdm(train_dataloader)):
            # print(data)
            bert_input , bert_input_for_next , bert_label, segment_label , last_gt =data

            bert_input = bert_input.to(args.device)
            bert_label = bert_label.to(args.device)
            bert_input_for_next = bert_input_for_next.to(args.device)
            segment_label = segment_label.to(args.device)
            last_gt = last_gt.to(args.device)

            # print(bert_input.size())
            # print(bert_label.size())
            # print(segment_label.size())


            output , _ = model(bert_input, segment_label) # B*100*5000
            # _ , last_pred = model(bert_input_for_next , segment_label)
            if idx < 5 :
                print(output.size())
                # print(last_pred.size())
                # print(last_gt)
                # print(bert_input_for_next[0])
                print(bert_input[0])
                print(segment_label[0])
                print(bert_label[0])


            output = output.reshape(output.shape[0] * output.shape[1], -1)

            loss=  criterion(output, bert_label.reshape(-1))
            # loss_last  = criterion(last_pred, last_gt)
            loss = loss

            equal = torch.argmax(output, dim=1) == bert_label.reshape(-1)
            # equal_last  = torch.argmax(last_pred , dim = 1) == last_gt
            # equal_last_num =  torch.sum(equal_last * (last_gt !=  0 ))
            right_num = torch.sum(equal * (bert_label.reshape(-1) != 0))
            masked_num = torch.sum(bert_label.reshape(-1) != 0)

            print(right_num, masked_num)
            print('loss is {}'.format(loss.item()))
            if idx % args.log_freq == 0:
               print("this is {}th {}batch time".format(epoch , idx ))

            if idx % 2000 == 0 :
                nsml.save(str(epoch) + "0" + str(idx))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        nsml.save(str(epoch)+ "end")
        model.eval()
        with torch.no_grad() :
            for idx, data in enumerate(tqdm(test_dataloader)):
                bert_input, bert_input_for_next , bert_label, segment_label, last_gt = data
                bert_input = bert_input.to(args.device)
                bert_label = bert_label.to(args.device)
                bert_input_for_next = bert_input_for_next.to(args.device)
                segment_label = segment_label.to(args.device)
                last_gt = last_gt.to(args.device)

                output , _ = model(bert_input,segment_label)  # B*100*5000
                # _ , last_pred  = model(bert_input_for_next , segment_label)
                output   = output.reshape(output.shape[0] * output.shape[1], -1)

                # loss = criterion(output, bert_label.reshape(-1))
                #
                # print('loss is {}'.format(loss.item()))
                if idx % 10 == 0:
                    print('valid_output_of_model')
                    equal = torch.argmax(output, dim=1) == bert_label.reshape(-1)
                    print(equal.size())
                    right_num = torch.sum(equal * (bert_label.reshape(-1) != 0))
                    masked_num = torch.sum(bert_label.reshape(-1) != 0)
                    print(right_num)
                    print(masked_num)

        nsml.save(str(epoch))

def bind_model(model, device):
    def save(dir_name):
        torch.save(model.state_dict(), dir_name + f'/params.pkl')

    def load(dir_name):
        params_fname = dir_name + f'/params.pkl'
        if device.type == 'cpu':
            state = torch.load(params_fname, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(params_fname)
        model.load_state_dict(state)

    def infer(x):
        """
        :params
            x (100,): Purchase history of a single user.
        """
        model.eval()

        x = x.astype(np.int64).reshape(1, -1)  # Append batch axis (100,) -> (1, 100)
        x = torch.as_tensor(x, device=device)

        pred = model(x).detach().cpu().numpy()[0]
        pred = np.argmax(pred)
        return pred

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    args.add_argument('--batch_size', type=int, default=256)
    args.add_argument('--output_dir', type=str, default='result')
    args.add_argument('--fc_dim', type=int, default=128)
    args.add_argument('--epoch_max', type=int, default=100)
    args.add_argument('--lr', type=float, default=0.0001)
    args.add_argument('--load_model', type=str, default=None)

    args.add_argument('--loss_function', type=str, default=None)  # custom.
    args.add_argument('--optimizer', type=str, default='adam')
    args.add_argument('--model', type=str, default='baseline')
    args.add_argument('--log_freq', type=int, default=50)
    args.add_argument('--device')
    args.add_argument('--valid_ratio', type = float , default= 0.1 )
    args.add_argument('--hidden_dim', type=int, default=768)
    args.add_argument('--n_layers', type=int, default=12)
    args.add_argument('--attn_heads', type=int, default=12)

    config = args.parse_args()


    print(config)
    # Hyperparameters & Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    epoch_max = config.epoch_max
    batch_size = config.batch_size
    init_lr = config.lr

    train_dataset_path = DATASET_PATH + '/train'
    all_x = np.load(train_dataset_path + '/train_data/all_x.npy').astype(np.int64)  # Pre-training data
    # model = BERTMASK(bert = BERT(vocab_size = 3110 , hidden = 384), shopping_size= 3110, all_x = all_x ).to(device)

    model = BERT_forall(bert =BERT(vocab_size = 5003, hidden = config.hidden_dim , n_layers= config.n_layers , attn_heads= config.attn_heads), shopping_size= 5003 , all_x = None ).to(device)

    # model = Bert4rec.BERT4Rec(num_layers = 6 ,num_heads= 6 , num_item= 5000 , num_user= 1200000 , dropout_rate = 0.1 , hidden_units = 384 , max_len = 100 , device = device)
    print(model)
    bind_model(model, device)


    if config.pause:
        nsml.paused(scope=locals())

    # Load data
    os.system("nvidia-smi")
    os.system("sudo rm -rf ~/.nv")
    print(torch.cuda.is_available())
    print('Load data...')
    train_dataset_path = DATASET_PATH + '/train'
    all_x = np.load(train_dataset_path + '/train_data/all_x.npy').astype(np.int64)  # Pre-training data
    train_x = np.load(train_dataset_path + '/train_data/x.npy').astype(np.int64)
    train_y = np.load(train_dataset_path + '/train_label').astype(np.int64)
    print('end load raw data')

    mydataset= BERTdataset(all_x = all_x )
    print('end making BERTdataset')
    train_bert(config ,model =model , dataset = mydataset )
