import util
from BERT import BERTClassifer, get_BERT , BERTMASK , BERT_forall
from dataset import BertDatatset_classification

import argparse
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
from sklearn.model_selection import KFold


from nsml import DATASET_PATH
import nsml
from datetime import datetime
VOCAB_SIZE = 5003
CLASS_NUM = 10
PAD = 0
MASK = 5001
CLR = 5002
EOS = 5003
NILL_IGNORE = 5002

# rand = torch.rand((50,50,786))
# rand = torch.mean(rand, dim =1 )
# print(rand.size())


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
        model.eval()
        x = x.astype(np.int64).reshape(1, -1)  # Append batch axis (100,) -> (1, 100)
        # print(x)
        # for BERT counting
        shop_len = np.count_nonzero(x)
        a = np.insert(x, 0, CLR)
        x = torch.as_tensor(a, device=device)
        segment_label = torch.as_tensor(np.zeros_like(a).astype(np.long), device=device)
        pred = model(x.unsqueeze(0),segment_label.unsqueeze(0)).detach().cpu().numpy()[0]
        print(pred)
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
    args.add_argument('--freeze_bert', type = bool , default = True )
    args.add_argument('--hidden_dim', type = int, default= 768)
    args.add_argument('--n_layers', type = int , default = 6 )
    args.add_argument('--attn_heads',  type = int  , default = 2 )
    args.add_argument("--data_aug" , type = bool ,default = False)
    args.add_argument("--k_fold" ,type = int , default = None)
    config = args.parse_args()
    args = config
    config.data_aug = False
    print(config)
    # Hyperparameters & Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    epoch_max = config.epoch_max
    batch_size = config.batch_size
    init_lr = config.lr

    model = BERTMASK(bert = get_BERT(args), shopping_size= VOCAB_SIZE ).to(device)

    if config.model == 'bert_all':
        VOCAB_SIZE = 5003
        model = BERT_forall(bert=get_BERT(args), shopping_size=VOCAB_SIZE).to(device)

    if config.load_model is not None:
        session, ckpt = config.load_model.split('_')
        bind_model(model, device)

        nsml.load(checkpoint=ckpt, session='KR96419/airush2022-2-5/' + session)
        bert = get_BERT(args)

        bert.load_state_dict(model.bert.state_dict())
        model = BERTClassifer ( bert = bert ,args = config, hidden = args.hidden_dim , freeze = args.freeze_bert).to(device)


        nsml.save(config.load_model + 'classifier')
        print('saved')

    bind_model(model, device)

    print(model)

    if config.pause:
        nsml.paused(scope=locals())

    os.system("nvidia-smi")
    os.system("pip install torch 1.15.0")
    print(torch.cuda.is_available())
    print('Load data...')
    train_dataset_path = DATASET_PATH + '/train'
    all_x = np.load(train_dataset_path + '/train_data/all_x.npy').astype(np.int64)  # Pre-training data
    train_x = np.load(train_dataset_path + '/train_data/x.npy').astype(np.int64)
    train_y = np.load(train_dataset_path + '/train_label').astype(np.int64)
    print('end load raw data')


    if config.k_fold is None :
        train_x, valid_x, train_y , valid_y  = train_test_split(train_x, train_y , test_size= args.valid_ratio , shuffle = True)
        if config.data_aug:
            train_x, train_y = util.data_aug(train_x, train_y)
        train_dataset = BertDatatset_classification(x=train_x, y=train_y, istrain=False , masking = True )
        valid_dataset = BertDatatset_classification(x=valid_x, y=valid_y, istrain=False)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)
    if config.data_aug:
        train_x, train_y = util.data_aug(train_x, train_y)
        print(len(train_x) , len(train_y))
    #     valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)
    # print(len(train_x) , len(valid_x))
    #


    # print(len(train_x), len(train_y))

    # dataset = BertDatatset_classification( x= train_x ,y = train_y)
    # train_dataset, valid_dataset = train_test_split(dataset, test_size=args.valid_ratio, shuffle=True)
    # print(len(train_dataset))


    criterion = util.SmoothCrossEntropyLoss(smoothing  = 0.1 ).to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay = 0.0001 )
    scheduler  =torch.optim.lr_scheduler.StepLR(optimizer = optimizer , step_size = 5 , gamma = 0.2)
    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))

    print('train_BERT_mtm_start')
    best = 0
    min_loss = 10000000.0
    for epoch in range(args.epoch_max):
            start = datetime.now()
            model.train()
        # if config.k_fold is not None:
        #     kf = KFold(n_splits=config.k_fold , shuffle= True )
        #
        # for set_num , (train_index, test_index) in enumerate(kf.split(train_x)):
        #     ...
        #     print("TRAIN:", train_index, "TEST:", test_index)
        #     x_train, x_valid = train_x[train_index], train_x[test_index]
        #     y_train, y_valid = train_y[train_index], train_y[test_index]
        #     train_dataloader = torch.utils.data.DataLoader(BertDatatset_classification( x= x_train ,y = y_train , istrain= True) , batch_size = config.batch_size , shuffle= True )
        #     valid_dataloader = torch.utils.data.DataLoader(BertDatatset_classification( x= x_valid ,y = y_valid , istrain= False) , batch_size = 128 , shuffle= True )

            for idx, data in enumerate(tqdm(train_dataloader)):
                # print(data)
                model.train()
                bert_input, bert_label, segment_label = data
                bert_input = bert_input.to(args.device)
                bert_label = bert_label.to(args.device)
                segment_label = segment_label.to(args.device)
                # print(bert_input.size())
                # print(bert_label.size())
                # print(segment_label.size())

                output = model(bert_input, segment_label)  # B*100*5000
                # print(output)
                # print(bert_label)
                loss = criterion(output, bert_label)
                if idx < 3 :
                    print(bert_input[0])

                print('loss is {}'.format(loss.item()))
                if idx % args.log_freq == 0:
                    print('train_output_of_model')
                    equal = torch.argmax(output, dim=1) == bert_label

                    print(bert_label*equal )
                    print(torch.sum(equal))
                    # print(bert_label.size(0))
                if idx % 200 == 0:
                    nsml.save(str(epoch) + "_" + str(idx))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                # model.eval()
                # with torch.no_grad():
                #     for idx2, data in enumerate(tqdm(valid_dataloader)):
                #         bert_input, bert_label, segment_label = data
                #         bert_input = bert_input.to(args.device)
                #         bert_label = bert_label.to(args.device)
                #         segment_label = segment_label.to(args.device)
                #         # print(bert_input.size())
                #         # print(bert_label.size())
                #         # print(segment_label.size())
                #
                #         output = model.forward(bert_input, segment_label)  # B*100*5000
                #
                #         loss = criterion(output, bert_label)
                #         #
                #         equal = torch.argmax(output, dim=1) == bert_label
                #
                #         if best < torch.sum(equal).item() :
                #             best = torch.sum(equal).item()
                #             nsml.save(str(epoch) +str(idx) + 'new_best')
            nsml.save(str(epoch) + "end")

            model.eval()
            loss_sum = 0.0
            with torch.no_grad() :
                for idx, data in enumerate(tqdm(valid_dataloader)):
                    bert_input, bert_label, segment_label = data
                    bert_input = bert_input.to(args.device)
                    bert_label = bert_label.to(args.device)
                    segment_label = segment_label.to(args.device)
                    # print(bert_input.size())
                    # print(bert_label.size())
                    # print(segment_label.size())

                    output = model.forward(bert_input,segment_label)  # B*100*5000

                    loss = criterion(output, bert_label)
                    #
                    print(' {} epoch loss is {}'.format(epoch , loss.item()))

                    print('{} valid_output_of_model'.format(epoch))
                    equal = torch.argmax(output, dim=1) == bert_label
                    print(bert_label)
                    print(bert_label * equal)
                    print(torch.sum(equal))
                    _, tk = torch.topk(output, 3, dim=1)
                    correct_pixels = torch.eq(bert_label[:, None, ...], tk).any(dim=1)
                    top_k_acc = torch.sum(correct_pixels)
                    top_k_num = top_k_acc.detach().item()
                    print(torch.max(torch.softmax(output , dim = 1 ), dim=1))
                    print(top_k_num/ bert_label.size(0))
                    print(bert_label.size(0))
                    loss_sum += loss.item()
                if (min_loss > loss_sum ) :
                    if config.k_fold is not None :

                        nsml.save(str(epoch)+"0" +str(set_num)+"best")
                    else :
                        nsml.save(str(epoch) + "0" + "best")
                    min_loss = loss_sum
            print("######################### {} #############################".format(datetime.now() - start))

            nsml.save(str(epoch))