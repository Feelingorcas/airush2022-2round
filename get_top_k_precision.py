import argparse
from BERT import BERTClassifer
from nsml import DATASET_PATH
import nsml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import MyDataset , BertDatatset_classification
from util import loss_func
from BERT import BERTMASK, BERT, get_vocab_dic, BERT_forall
from tqdm import tqdm

VOCAB_SIZE = 5004
CLASS_NUM = 10
PAD = 0
MASK = 5001
CLR = 5002
EOS = 5003
NILL_IGNORE = 5005


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

        # x = x.astype(np.int64).reshape(1, -1)  # Append batch axis (100,) -> (1, 100)
        # print(x)

        # for BERT counting
        shop_len = np.count_nonzero(x)
        a = np.insert(x, 0, CLR)
        a = np.insert(a, shop_len + 1, EOS)
        x = torch.as_tensor(a.astype(np.long), device=device)

        segment_label = np.ones_like(a[:shop_len + 2])
        segment_label = torch.as_tensor(
            np.append(segment_label, np.array([0 for i in range(len(a) - shop_len - 2)])).astype(np.long),
            device=device)

        pred = model(x.unsqueeze(0), segment_label.unsqueeze(0)).detach().cpu().numpy()[0]
        pred = np.argmax(pred)
        return pred

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    args.add_argument('--batch_size', type=int, default=512)
    args.add_argument('--output_dir', type=str, default='result')
    args.add_argument('--fc_dim', type=int, default=128)
    args.add_argument('--epoch_max', type=int, default=20)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--load_model', type=str, default=None)

    args.add_argument('--loss_function', type=str, default=None)  # custom.
    args.add_argument('--optimizer', type=str, default='adam')
    args.add_argument('--model', type=str, default='baseline')
    args.add_argument('--log_freq', type=int, default=100)
    args.add_argument('--device')
    args.add_argument('--valid_ratio', type=float, default=0.1)
    args.add_argument('--freeze_bert', type=bool, default=True)
    config = args.parse_args()

    # Hyperparameters & Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch_max = config.epoch_max
    batch_size = config.batch_size
    init_lr = config.lr
    args.device = device
    # Bind model
    model = MyModel().to(device=device)
    if config.model == 'bert':
        model = BERTClassifer(bert=BERT(vocab_size=VOCAB_SIZE)).to(device)
    elif config.model == 'bert_all':
        VOCAB_SIZE = 5005
        model = BERTClassifer(bert=BERT(vocab_size=VOCAB_SIZE)).to(device)
    bind_model(model, device)

    if config.load_model is not None:
        session, ckpt = config.load_model.split('_')
        nsml.load(checkpoint=ckpt, session='KR96419/airush2022-2-5/' + session)
        nsml.save(config.load_model + 'loaded')
        print('saved')

    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    if config.pause:
        nsml.paused(scope=locals())

    # Load data
    print('Load data...')
    train_dataset_path = DATASET_PATH + '/train'
    all_x = np.load(train_dataset_path + '/train_data/all_x.npy').astype(np.int64)  # Pre-training data
    train_x = np.load(train_dataset_path + '/train_data/x.npy').astype(np.int64)
    train_y = np.load(train_dataset_path + '/train_label').astype(np.int64)
    #

    train_dataset = BertDatatset_classification(train_x, train_y , istrain= False)

    # Train
    model.train()
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    cnt = 0
    loss_sum = 0
    epoch = 1
    top_k_num = 0
    with torch.no_grad():
        for idx, data in enumerate(tqdm(train_loader)):
            bert_input, bert_label, segment_label = data
            bert_input = bert_input.to(args.device)
            bert_label = bert_label.to(args.device)
            segment_label = segment_label.to(args.device)
            # print(bert_input.size())
            # print(bert_label.size())
            # print(segment_label.size())

            output = model.forward(bert_input, segment_label)  # B*100*5000
            _, tk = torch.topk(output, 3, dim=1)
            correct_pixels = torch.eq(bert_label[:, None, ...], tk).any(dim=1)
            print(tk)
            print(bert_label)
            print(correct_pixels)

            top_k_acc = torch.sum(correct_pixels)
            top_k_num += top_k_acc.detach().item()
            print(top_k_acc)
            equal = torch.argmax(output, dim=1) == bert_label
            print(bert_label)
            print(bert_label * equal)
            print((torch.sum(segment_label ,dim= 1 )*equal).float().mean())
            print(torch.max(output ,dim = 1 ))
            print(torch.sum(equal))

            print(bert_label.size(0))

    print(top_k_num)
    nsml.save(str(epoch))
