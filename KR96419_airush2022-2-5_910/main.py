
import argparse
from BERT import BERTClassifer , get_BERT
from nsml import DATASET_PATH
import nsml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import MyDataset , BertDatatset_classification
from util import loss_func , data_aug
from BERT import BERTMASK , BERT ,get_vocab_dic , BERT_forall ,BERTessemble
from tqdm import tqdm
VOCAB_SIZE = 5004
CLASS_NUM = 10
PAD = 0
MASK = 5001
CLR = 5002
EOS = 5003
NILL_IGNORE = 5005
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from MPL_pytorch.models import   ModelEMA

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

        a = np.insert(x, 0, CLR)
        # a = np.insert(a, shop_len + 1, EOS)
        x = torch.as_tensor(a.astype(np.long), device=device)

        # segment_label = np.ones_like(a[:shop_len + 2])
        # segment_label = torch.as_tensor(np.append(segment_label, np.array([0 for i in range(len(a) - shop_len - 2)])).astype(np.long), device = device)
        segment_label = torch.as_tensor(np.zeros_like(a).astype(np.long) , device = device )


        ## TTA
        # for i in range(1,shop_len+2) :
        #     prob = random.random()
        #     if prob < 0.2 :
        #         a[i] = random.randint(1, 5000)
        # x_tta = torch.as_tensor(a.astype(np.long) , device =  device)
        #
        # x = torch.stack((x,x_tta) , dim = 0 )

        # print(x.size())
        # print(segment_label.expand(2,segment_label.size(0)))
        pred = model(x.unsqueeze(0),segment_label.unsqueeze(0)).detach().cpu().numpy()[0]
        print(pred)
        # pred = torch.sum(torch.softmax(model(x, segment_label.expand(2,segment_label.size(0))) , dim = 1)  , dim = 0).detach().cpu().numpy()
        # # print(pred)
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
        args.add_argument('--log_freq' ,type = int ,default  = 100)
        args.add_argument('--device')
        args.add_argument('--valid_ratio', type=float, default=0.1)
        args.add_argument('--freeze_bert', type=bool, default=True)
        args.add_argument('--hidden_dim', type=int, default=768)
        args.add_argument('--n_layers', type=int, default=6)
        args.add_argument('--attn_heads', type=int, default=2)
        args.add_argument('--essemble' , type = int , default = 1 )
        config = args.parse_args()
        args =config

        # Hyperparameters & Settings
        device = torch.device('  cuda' if torch.cuda.is_available() else 'cpu')
        args.device = device
        epoch_max = config.epoch_max
        batch_size = config.batch_size
        init_lr = config.lr


        # essemble 용 코드

        VOCAB_SIZE = 5003
        # Bind model
        # model = MyModel().to(device=device)
        if config.model  == 'bert'  :
            model = BERTClassifer(bert=BERT(vocab_size=VOCAB_SIZE , hidden = config.hidden_dim  , attn_heads= config.attn_heads, n_layers= config.n_layers) ,args = config,  hidden = config.hidden_dim).to(device)
        elif config.model == 'bert_all'  :
            VOCAB_SIZE = 5003
            model = BERTClassifer(
                bert=BERT(vocab_size=VOCAB_SIZE, hidden=config.hidden_dim, attn_heads=config.attn_heads,
                          n_layers=config.n_layers), args=config, hidden=config.hidden_dim).to(device)
        elif config.model == 'EDA' :
            model = BERTClassifer(
                bert=BERT(vocab_size=VOCAB_SIZE, hidden=config.hidden_dim, attn_heads=config.attn_heads,
                          n_layers=config.n_layers), args=config, hidden=config.hidden_dim).to(device)

            model = ModelEMA(model = model)
            bind_model(model, device)
            if config.load_model is not None:

                a = config.load_model.split('_')
                if len(a) > 2 :
                    session = a[0]
                    ckpt = a[1]+'_'+a[2]
                else :
                    session = a[0]
                    ckpt = a[1]
                nsml.load(checkpoint=ckpt, session='KR96419/airush2022-2-5/' + session)
                nsml.save(config.load_model)

        print('saved')



        if config.pause:
            nsml.paused(scope=locals())
        #
        # ######################### load ########################
        # if config.essemble > 1 :
        #     all = config.load_model.split('*')
        #     a = all[0]
        #     a = a.split('_')
        #     if len(a) > 2:
        #         session = a[0]
        #         ckpt = a[1] + '_' + a[2]
        #     else:
        #         session = a[0]
        #         ckpt = a[1]
        #     nsml.load(checkpoint=ckpt, session='KR96419/airush2022-2-5/' + session)
        #
        #
        #     a = all[1]
        #     a = a.split('_')
        #     if len(a) > 2:
        #         session = a[0]
        #         ckpt = a[1] + '_' + a[2]
        #     else:
        #         session = a[0]
        #         ckpt = a[1]
        #
        #     model2 = BERTClassifer(bert=BERT(vocab_size=VOCAB_SIZE), args=config, hidden=config.hidden_dim).to(device)
        #     bind_model(model2, device)
        #
        #     nsml.load(checkpoint=ckpt, session='KR96419/airush2022-2-5/' + session)
        #     model2 = BERTClassifer(bert=BERT(vocab_size=VOCAB_SIZE),args = config,  hidden = config.hidden_dim ).to(device)
        #     model_essemble = BERTessemble(model ,model2)
        #     bind_model(model_essemble, device)
        #     nsml.save('essemble')
        #
        #     print('essemble ok ')
        # else :
        #     if config.load_model is not None:
        #
        #         a = config.load_model.split('_')
        #         if len(a) > 2 :
        #             session = a[0]
        #             ckpt = a[1]+'_'+a[2]
        #         else :
        #             session = a[0]
        #             ckpt = a[1]
        #         nsml.load(checkpoint=ckpt, session='KR96419/airush2022-2-5/' + session)
        #         nsml.save(config.load_model)
        #
        # print('saved')

        #######################################################################################################
        if config.model == 'bert_all':
            VOCAB_SIZE = 5003
            model = BERT_forall(bert=get_BERT(args), shopping_size=VOCAB_SIZE).to(device)

        if config.load_model is not None:
            session, ckpt = config.load_model.split('_')
            bind_model(model, device)

            nsml.load(checkpoint=ckpt, session='KR96419/airush2022-2-5/' + session)
            bert = get_BERT(args)

            bert.load_state_dict(model.bert.state_dict())
            model = BERTClassifer(bert=bert, args=config, hidden=args.hidden_dim, freeze=args.freeze_bert).to(device)

            nsml.save(config.load_model + 'classifier')
            print('saved')

        bind_model(model, device)

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

        dataset = BertDatatset_classification(x=train_x, y=train_y , istrain= False)
        # new_train_x , new_train_y  = data_aug(train_x, train_y)
        # print(len(new_train_x))
        dataloader  = torch.utils.data.DataLoader(dataset, batch_size = 1 , shuffle = False)
        print('Loading data is finished!')


        # model for embedding
        bert = model.bert.to(device)

        embeddings = [ ]
        inputs  =  [ ]
        labels =  [ ]

        for idx , data in enumerate(dataloader) :

            bert_input, bert_label, segment_label = data

            bert_input = bert_input.to(args.device)
            bert_label = bert_label.to(args.device)
            segment_label = segment_label.to(args.device)

            embedding = bert(bert_input, segment_label)
            print(embedding.size())
            embedding= torch.mean(embedding[:,1:,:]  , dim = 1 )
            print(embedding.squeeze().size())
            if idx < 5 :
                print(bert_input)
                print(bert_label)
            embeddings.append(embedding.squeeze().detach().cpu().numpy())
            labels.append(int(bert_label.squeeze().detach().item()))
            inputs.append(bert_input.squeeze().detach().cpu().numpy())


        scaler = MinMaxScaler()
        data_scale = scaler.fit_transform(embeddings)
        model = KMeans(n_clusters= 10 , random_state=7914)
        model.fit(embeddings)
        a = model.fit_predict(embedding)

        for embedding  , label in zip(embeddings, labels) :
                print(model(embedding) , label)

        # determined ={}
        # for i in range(1, 101) :
        #     for x,y  in zip(train_x, train_y) :
        #         if np.count_nonzero(x)  == 0 :
        #             determined[x[0]] = y
        #         elif np.count_nonzero(x) == i :
        #
        #             exist = [ ]
        #             not_exist  = [ ]
        #             for ele in x :
        #                 if ele in determined.keys :
        #                     exist.append(determined[ele])
        #                 else :
        #                     not_exist.append(ele)
        #             if len(exist)  == i-1 and np.all(exist == exist[0]) :
        #                 determined[x[0]]
