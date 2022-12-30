import sys
sys.path.append('./GCL2')
#
import torch
import os.path as osp
import GCL2.losses as L
import GCL2.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam

# from GCL.eval import get_split, LREvaluator_mia

# from GCL2.eval import get_split
# from GCL2.eval.logistic_regression import LREvaluator

from GCL2.models import DualBranchContrast_mia
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid,Amazon
import numpy as np
import networkx as nx
# from . import train_test_split
# from . import eval
#
# from . import logistic_regression

from GCL2 import preprocessing


import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score,accuracy_score

from GCL2.eval import BaseEvaluator

import pandas as pd

import random
import os
import pickle as pk

import itertools

def readedges2(file_name):
    file = open(file_name)

    dataMat = []
    for line in file.readlines():
        curLine = line.strip().split('\t')
        # print('111',curLine)
        if curLine==['']:
            dataMat.append([])
        else:
            floatLine = list(map(int, curLine))
        # print(floatLine)
            dataMat.append(floatLine)

    # embeddings = np.array(dataMat,dtype='int')

    return dataMat


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator_mia(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')
                    test_acc=accuracy_score(y_test, y_pred)

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro,'Acc':test_acc})
                    pbar.update(self.test_interval)

        output_train = classifier(x).detach().cpu().numpy()

        return  {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro,
            'acc':test_acc,
            'output_train':output_train
        }




def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }



class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index0, edge_index,edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        # print(x, x1, x2)
        # print(edge_index, edge_index1, edge_index2)
        #
        # print(x.size(), x1.size(), x2.size())
        # print(edge_index.size(), edge_index1.size(), edge_index2.size())
        #
        # exit()

        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2,edge_index1,edge_index2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data,edges_train_index, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2,edge_index1,edge_index2 = encoder_model(data.x,data.edge_index,edges_train_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    # print('KKKKK',h1,h1.size(),h1.type())
    # exit()
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item(),edge_index1,edge_index2

def train_mia(encoder_model, contrast_model, data,edges_train_index, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2,edge_index1,edge_index2 = encoder_model(data.x,data.edge_index,edges_train_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    # print('KKKKK',h1,h1.size(),h1.type())
    # exit()
    loss,l1,l2 = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item(),edge_index1,edge_index2,l1,l2,z, z1, z2



def test(encoder_model, data,edges_train_index):
    encoder_model.eval()
    z, z1, z2,_,_ = encoder_model(data.x,data.edge_index,edges_train_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result= LREvaluator_mia()(z, data.y, split)

    print(result)


    return result,z, z1, z2


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device(f'cuda:{os.environ["CUDA_VISIBLE_DEVICES"]}')
    # torch.cuda.set_device(1)
    # device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dt='Citeseer'
    dataset = Planetoid(path, name=dt, transform=T.NormalizeFeatures())###Cora, Citeseer
    # dataset = Amazon(path, dt, pre_transform=None)###Computers, Photo

    data = dataset[0].to(device)

    ratio=0.2

    aug1 = A.Compose([A.EdgeRemoving(pe=ratio), A.FeatureMasking(pf=0.2)])
    aug2 = A.Compose([A.EdgeRemoving(pe=ratio), A.FeatureMasking(pf=0.2)])

    # print(data.edge_index)
    # print(data.edge_index.type())
    # exit()

    edge_index0_all_oneside = []
    edge_index0_all = []

    edge_index0 = data.edge_index.detach().cpu().numpy()
    edge_index0 = edge_index0.transpose()
    for ed in edge_index0:
        if ed[0] > ed[1]:
            edge_index0_all.append([ed[0], ed[1]])
            continue
        else:
            edge_index0_all.append([ed[0], ed[1]])
            edge_index0_all_oneside.append([ed[0], ed[1]])
    edge_index0_all_oneside = np.array(edge_index0_all_oneside)
    edge_index0_all = np.array(edge_index0_all)

    g = nx.Graph()
    g.add_edges_from(edge_index0_all)
    adj_sparse = nx.to_scipy_sparse_matrix(g)
    random.seed(42)
    train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)

    res_dir = '%s-grace-mia-white-2-%s' % (dt,ratio)

    with open('./%s/%s-train_test_split' % (res_dir, dt), 'wb') as f:
        pk.dump(train_test_split, f)

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split
    # print(adj_train)
    g_train0 = nx.from_scipy_sparse_matrix(
        adj_train)  # new graph object with only non-hidden edges, keep all the original nodes

    edge_tuples0 = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in g_train0.edges()]
    # print(edge_tuples0)

    train_edges0 = set(edge_tuples0)  # initialize train_edges to have all edges
    train_edges0 = np.array([list(edge_tuple) for edge_tuple in train_edges0])
    # print(train_edges1)


    edge_tuples_test0 = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in test_edges]

    edges_test0 = set(edge_tuples_test0)  # initialize test_edges to have all edges
    edges_test0 = np.array([list(edge_tuple) for edge_tuple in edges_test0])



    out = open('%s/%s-edges-train.txt' % (res_dir, dt), 'w')
    for item in train_edges0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test.txt' % (res_dir, dt), 'w')
    for item in edges_test0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    # adj = adj_train
    #
    # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    train_edges_1=np.concatenate((train_edges0[:,1].reshape(-1,1),train_edges0[:,0].reshape(-1,1)),axis=1)
    train_edges_1=np.transpose(np.array(train_edges_1))
    train_edges_2 = np.transpose(np.array(train_edges0))
    # loop_nodes=np.arange(0,g.number_of_nodes())
    # train_edges_3=np.concatenate((loop_nodes.reshape(-1,1),loop_nodes.reshape(-1,1)),axis=1)
    # train_edges_3 = np.transpose(np.array(train_edges_3))

    edges_train_index=np.concatenate((train_edges_1,train_edges_2),axis=1)


    edges_train_index = torch.from_numpy(np.array(edges_train_index)).long().to(device)


    gconv = GConv(input_dim=dataset.num_features, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=32).to(device)
    contrast_model = DualBranchContrast_mia(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    encoder_weights0 = []
    encoder_weights1 = []

    fc_weights0 = []
    fc_weights1 = []
    losss1=[]
    losss2=[]

    z1_trains=[]
    z2_trains=[]

    edge_index1_trains=[]
    edge_index2_trains = []


    best_valid_loss=99999999

    with tqdm(total=10000, desc='(T)') as pbar:
        for epoch in range(1, 10001):
            loss,edge_index1,edge_index2,loss1,loss2,z_train, z1_train, z2_train = train_mia(encoder_model, contrast_model, data,edges_train_index, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

            patience = 100

            if loss < best_valid_loss:
                best_valid_loss = loss
                trail_count = 0
                best_epoch = epoch
                torch.save(encoder_model.state_dict(), os.path.join('./checkpoint',
                                                                    'tmp',
                                                                    f'grace_{dt}_best.pt'))

                z1_trains.append(z1_train.detach().cpu().numpy())
                z2_trains.append(z2_train.detach().cpu().numpy())

                # print(edge_index1.detach().cpu().numpy())
                # print(edge_index1.size())

                edge_index1_trains.append(edge_index1.detach().cpu().numpy())
                edge_index2_trains.append(edge_index2.detach().cpu().numpy())


            else:
                trail_count += 1

                if trail_count > patience:
                    print(f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                    break

                else:
                    edge_index1_trains.append(edge_index1.detach().cpu().numpy())
                    edge_index2_trains.append(edge_index2.detach().cpu().numpy())

                    z1_trains.append(z1_train.detach().cpu().numpy())
                    z2_trains.append(z2_train.detach().cpu().numpy())

        encoder_model.load_state_dict(torch.load(os.path.join('./checkpoint', 'tmp',
                                                              f'grace_{dt}_best.pt')))
        para = {}
        cnt = 0

        for p in encoder_model.parameters():
            # print(p)
            p = p.cpu().detach().numpy()
            # print(p)
            para[cnt] = p
            cnt += 1

        encoder_weight0 = p[1]
        encoder_weight1 = p[3]

        fc_weight0 = p[4]
        fc_weight1 = p[6]

        encoder_weights0.append(encoder_weight0)
        encoder_weights1.append(encoder_weight1)

        fc_weights0.append(fc_weight0)
        fc_weights1.append(fc_weight1)

        losss1.append(loss1.cpu().detach().numpy())
        losss2.append(loss2.cpu().detach().numpy())

    test_result,z, z1, z2 = test(encoder_model, data,edges_train_index)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f},Acc={test_result["acc"]:.4f}')

    output_train=test_result["output_train"]


    edge_index1 = edge_index1.detach().cpu().numpy()
    edge_index2 = edge_index2.detach().cpu().numpy()

    # print(edge_index0, edge_index1, edge_index2, edges_test)

    print(np.shape(edge_index0), np.shape(edge_index1), np.shape(edge_index2))

    edges1 = []
    edges2 = []

    edges1_idx = []
    edges2_idx = []

    for i in range(np.shape(edge_index1)[1]):
        edges1.append([edge_index1[0][i], edge_index1[1][i]])
        edges1_idx.append(edge_index1[0][i] * np.shape(data.x)[0] + edge_index1[1][i])

    for i in range(np.shape(edge_index2)[1]):
        edges2.append([edge_index2[0][i], edge_index2[1][i]])
        edges2_idx.append(edge_index2[0][i] * np.shape(data.x)[0] + edge_index2[1][i])



    for name, param in encoder_model.named_parameters():
        print(name,param.size())

    for name, param in contrast_model.named_parameters():
        print('***',name,param.size())

    # encoder.layers.0.bias torch.Size([32])
    # encoder.layers.0.lin.weight torch.Size([32, 1433])
    # encoder.layers.1.bias torch.Size([32])
    # encoder.layers.1.lin.weight torch.Size([32, 32])
    # fc1.weight torch.Size([32, 32])
    # fc1.bias torch.Size([32])
    # fc2.weight torch.Size([32, 32])
    # fc2.bias torch.Size([32])
    # exit()

    print(np.shape(z1_trains),np.shape(z2_trains))

    edges_train_all=train_edges0

    emb_matrix0 = z.detach().cpu().numpy()
    emb_matrix1=z1.detach().cpu().numpy()
    emb_matrix2 = z2.detach().cpu().numpy()

    z1_trains_=np.array(z1_trains)
    z2_trains_=np.array(z2_trains)

    with open('./%s/%s-aug1.pkl' % (res_dir, dt), 'wb') as f:
        pk.dump(edge_index1_trains, f)

    with open('./%s/%s-aug2.pkl' % (res_dir, dt), 'wb') as f:
        pk.dump(edge_index2_trains, f)

    with open('./%s/%s-aug1-embed.pkl' % (res_dir, dt), 'wb') as f:
        pk.dump(z1_trains_, f)

    with open('./%s/%s-aug2-embed.pkl' % (res_dir, dt), 'wb') as f:
        pk.dump(z2_trains_, f)


    return emb_matrix0,output_train,edges_train_all,edges_test0,data.y,res_dir,z1_trains,z2_trains,dt,edge_index1_trains,edge_index2_trains,z1_trains_,z2_trains_


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    # print(h1,h1.type())
    # h1 = F.normalize(h1)
    # h2 = F.normalize(h2)
    # h1=torch.add_(h1,0.000000001)
    return h1 @ h2.t()

def get_edge_embeddings(edge_list, emb_matrixs,idx_epoches_all ):
    embs = []
    i=0
    for edge in edge_list:
        node1 = int(edge[0])
        node2 = int(edge[1])
        emb=[]
        # print(i)
        # print(idx_epoches_all[i,:])
        # print(len(idx_epoches_all[i,:]))
        for emb_matrix in emb_matrixs[idx_epoches_all[i,:],:,:]:
            emb1 = emb_matrix[node1]
            #print(np.shape(emb1))
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            sim1 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)+0.0000000000000000000000000000001)

            sim2 = np.dot(emb1, emb2)

            sim3 = np.linalg.norm(np.array(emb1) - np.array(emb2))

            #edge_emb = np.array(emb1) + np.array(emb2)
            # print(np.shape(edge_emb))
            emb.append(sim1)
            emb.append(sim2)
        i+=1
        embs.append(emb)
    embs = np.array(embs)
    return embs

def get_edge_posts(edge_list,train_preds):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        pre1 = train_preds[node1]
        #print(np.shape(emb1))
        pre2 = train_preds[node2]

        pre_idx1 = np.argmax(pre1)
        pre_idx2 = np.argmax(pre2)
        train_pres_temp1 = np.sort(pre1)
        train_pres_temp2 = np.sort(pre2)
        if pre_idx1 == label[node1]:
            corr = 1
        else:
            corr = 0

        train_pres1_=([train_pres_temp1[-1], train_pres_temp1[-2], corr])

        if pre_idx2 == label[node2]:
            corr = 1
        else:
            corr = 0
        train_pres2_ = ([train_pres_temp2[-1], train_pres_temp2[-2], corr])

        edge_emb = np.multiply(train_pres1_, train_pres2_)
        #edge_emb = np.array(emb1) + np.array(emb2)
        print(np.shape(edge_emb))

        emb1=train_pres1_
        emb2 = train_pres2_

        sim1 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        sim2 = np.dot(emb1, emb2)

        sim3 = np.linalg.norm(np.array(emb1) - np.array(emb2))


        embs.append([sim1,sim2])
    embs = np.array(embs)

    return embs







if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    emb_matrix, output_train,edges_train_all,edges_test_all,label,res_dir,z1_trains,z2_trains,dt,aug1s,aug2s,aug1s_embed,aug2s_embed=main()

    # edges_train_inter=np.array(edges_train_inter)
    edges_train_all = np.array(edges_train_all)
    edges_test_all = np.array(edges_test_all)

    train_preds = output_train

    train_range1 = list(np.arange(np.shape(edges_train_all)[0]))

    # Train-set edge embeddings
    train_preds_sampled_idx1 = np.array(random.sample(train_range1, np.shape(edges_test_all)[0]))

    print(train_preds_sampled_idx1)

    train_edges_sampled1 = np.array(edges_train_all)[train_preds_sampled_idx1, :]



    out = open('%s/%s-edges-train_sampled.txt' % (res_dir, dt), 'w')
    for item in train_edges_sampled1:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test_sampled.txt' % (res_dir, dt), 'w')
    for item in edges_test_all:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    ylabel = [1] * len(train_preds_sampled_idx1) + [0] * len(train_preds_sampled_idx1)

    train_edges_list = train_edges_sampled1
    test_edges_list = np.array(edges_test_all)

    edges_list = np.concatenate((train_edges_list, test_edges_list), axis=0)

    ylabel1 = ylabel
    ylable1 = np.reshape(len(ylabel1), 1)
    y_label = np.zeros((np.shape(edges_list)[0], 3))
    for i in range(np.shape(edges_list)[0]):
        y_label[i][0] = edges_list[i][0]
        y_label[i][1] = edges_list[i][1]
        y_label[i][2] = ylabel[i]
    print(np.shape(y_label))

    y_label_train = np.zeros((np.shape(train_edges_list)[0], 3))
    for i in range(np.shape(train_edges_list)[0]):
        y_label_train[i][0] = train_edges_list[i][0]
        y_label_train[i][1] = train_edges_list[i][1]
        y_label_train[i][2] = 1
    print(np.shape(y_label_train))

    y_label_test = np.zeros((np.shape(test_edges_list)[0], 3))
    for i in range(np.shape(test_edges_list)[0]):
        y_label_test[i][0] = test_edges_list[i][0]
        y_label_test[i][1] = test_edges_list[i][1]
        y_label_test[i][2] = 0
    print(np.shape(y_label_test))

    sam_list_idx = list(range(np.shape(y_label_train)[0]))

    sam_list_idx_train = np.array(random.sample(sam_list_idx, int(0.3 * len(sam_list_idx))))

    sam_list_idx = list(range(np.shape(y_label_test)[0]))

    sam_list_idx_test = np.array(random.sample(sam_list_idx, int(0.3 * len(sam_list_idx))))

    y_test = np.concatenate((y_label_train[sam_list_idx_train], y_label_test[sam_list_idx_test]), axis=0)

    edges_mia = y_test

    edges_mia0 = np.array(edges_mia)[:, 0:2]

    edges_mia = np.array(edges_mia)
    index_pos = np.where(edges_mia[:, 2] == 1)[0]
    index_neg = np.where(edges_mia[:, 2] == 0)[0]

    print(len(index_pos), len(index_neg))

    edges_mia_pos0 = edges_mia[index_pos]
    edges_mia_neg0 = edges_mia[index_neg]

    edges_mia_pos = [[min(edge[0], edge[1]), max(edge[0], edge[1])] for edge in edges_mia_pos0]
    print(np.shape(edges_mia_pos))
    edges_mia_pos_idx = np.array(edges_mia_pos)[:, 0] * 99999 + np.array(edges_mia_pos)[:, 1]

    edges_mia_neg = [[min(edge[0], edge[1]), max(edge[0], edge[1])] for edge in edges_mia_neg0]

    edges_mia_neg_idx = np.array(edges_mia_neg)[:, 0] * 99999 + np.array(edges_mia_neg)[:, 1]

    train_edges_sampled_ = [[min(edge[0], edge[1]), max(edge[0], edge[1])] for edge in train_edges_sampled1]
    test_edges_sampled_ = [[min(edge[0], edge[1]), max(edge[0], edge[1])] for edge in edges_test_all]

    train_edges_sampled_idx = np.array(train_edges_sampled_)[:, 0] * 99999 + np.array(train_edges_sampled_)[:, 1]
    test_edges_sampled_idx = np.array(test_edges_sampled_)[:, 0] * 99999 + np.array(test_edges_sampled_)[:, 1]

    train_edges_pos_idx = np.setdiff1d(train_edges_sampled_idx, edges_mia_pos_idx)
    train_edges_neg_idx = np.setdiff1d(test_edges_sampled_idx, edges_mia_neg_idx)

    print(len(train_edges_sampled_idx), len(test_edges_sampled_idx), len(train_edges_pos_idx), len(train_edges_neg_idx))
    print(len(train_edges_pos_idx), len(train_edges_neg_idx))

    results = []

    aug1s_idx = []
    for aug in aug1s:
        # print(aug,np.shape(aug))
        aug = aug.T
        aug_ = [[min(edge[0], edge[1]), max(edge[0], edge[1])] for edge in aug]
        aug_idx = np.array(aug_)[:, 0] * 99999 + np.array(aug_)[:, 1]
        # print('$$$$$$$',np.shape(aug_idx))
        aug1s_idx.append(aug_idx)

    aug2s_idx = []
    for aug in aug2s:
        aug = aug.T
        aug_ = [[min(edge[0], edge[1]), max(edge[0], edge[1])] for edge in aug]
        aug_idx = np.array(aug_)[:, 0] * 99999 + np.array(aug_)[:, 1]
        # print('$$$$$$$', np.shape(aug_idx))
        aug2s_idx.append(aug_idx)

    #
    drop1s_pos_idx = []
    drop2s_pos_idx = []

    for aug_idx in aug1s_idx:
        drop_idx = np.setdiff1d(train_edges_pos_idx, aug_idx)
        drop1s_pos_idx.append(drop_idx)

    for aug_idx in aug2s_idx:
        drop_idx = np.setdiff1d(train_edges_pos_idx, aug_idx)
        drop2s_pos_idx.append(drop_idx)

    # print(drop1s_pos_idx)
    # print(drop2s_pos_idx)


    with open('./%s/%s-drop1s_pos_idx.txt' % (res_dir, dt), 'w') as f:
        for item in drop1s_pos_idx:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    with open('./%s/%s-drop2s_pos_idx.txt' % (res_dir, dt), 'w') as f:
        for item in drop2s_pos_idx:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    file_name = './%s/%s-drop1s_pos_idx.txt' % (res_dir, dt)
    drop1s_pos_idx0 = readedges2(file_name)
    # print(drop1s_pos_idx)

    file_name = './%s/%s-drop2s_pos_idx.txt' % (res_dir, dt)
    drop2s_pos_idx0 = readedges2(file_name)

    iterations = np.shape(drop2s_pos_idx0)[0]

    # iter_ratios=[0.2,0.4,0.6,0.8,1]
    iter_ratios = [1]

    # results=[]
    for iters in iter_ratios:
        iter_ = int(iterations * iters) - 1

        drop1s_pos_idx = drop1s_pos_idx0[0:iter_]
        drop2s_pos_idx = drop2s_pos_idx0[0:iter_]

        drop1s_pos_idx_ = list(itertools.chain.from_iterable(drop1s_pos_idx))
        drop2s_pos_idx_ = list(itertools.chain.from_iterable(drop2s_pos_idx))

        print(len(drop1s_pos_idx_), len(drop2s_pos_idx_))
        set1 = list(set(drop1s_pos_idx_))
        set2 = list(set(drop2s_pos_idx_))
        print(len(set1), len(set2))
        set0 = list(set(set1 + set2))
        # print(set0)
        print(len(set0))
        print(np.shape(edges_test_all)[0])
        # exit()
        idx_dic1 = dict()
        idx_dic2 = dict()
        idx_dic1_ = dict()
        idx_dic2_ = dict()
        for idx in set0:
            idx_dic1[idx] = 0
            idx_dic2[idx] = 0
            idx_dic1_[idx] = []
            idx_dic2_[idx] = []

        i = 0
        for idx in drop1s_pos_idx:
            for j in idx:
                idx_dic1[j] += 1
                idx_dic1_[j].append(i)
            i += 1

        i = 0
        for idx in drop2s_pos_idx:
            for j in idx:
                idx_dic2[j] += 1
                idx_dic2_[j].append(i)
            i += 1

        print(min(idx_dic1.values()), max(idx_dic1.values()))
        print(min(idx_dic2.values()), max(idx_dic2.values()))

        # print(idx_dic1,idx_dic2)
        idx_dic0 = []
        for idx in set0:
            idx_dic0.append(idx_dic1[idx] + idx_dic2[idx])
        # print(idx_dic0)
        print(min(idx_dic0), max(idx_dic0))

        train_edges_pos = []
        train_edges_neg = []
        for i in train_edges_pos_idx:
            node1 = int(i / 99999)
            node2 = i % 99999
            train_edges_pos.append([node1, node2])

        for i in train_edges_neg_idx:
            node1 = int(i / 99999)
            node2 = i % 99999
            train_edges_neg.append([node1, node2])

        test_edges_pos = np.array(edges_mia_pos)
        test_edges_neg = np.array(edges_mia_neg)

        epoches = np.shape(aug1s_embed)[0]
        idx_epoches = list(range(epoches))

        idx_epoches_all = []
        drop_idx_all = []

        for i in train_edges_pos_idx:

            if i in idx_dic1_.keys():  ###drop index

                drop_idx = idx_dic1_[i]
                # drop_idx_all.append(drop_idx)
                idx_epoches_ = list(set(idx_epoches).difference(set(drop_idx)))
                if len(drop_idx) < max(idx_dic1.values()):
                    # print(epoches,max(idx_dic1.values()),len(drop_idx))
                    # print(epoches-max(idx_dic1.values()) - len(drop_idx))
                    drop_idx_sample2 = random.sample(idx_epoches_, (epoches - max(idx_dic1.values()) - len(drop_idx)))
                    drop_idx_sample = random.sample(idx_epoches_, (max(idx_dic1.values()) - len(drop_idx)))
                    idx_epoches_ = list(set(idx_epoches_).difference(set(drop_idx_sample)))

                    drop_idx_ = list(drop_idx) + drop_idx_sample2

                    # print('111', len(drop_idx_))

                else:
                    idx_epoches_ = list(set(idx_epoches_))
                    drop_idx_ = idx_epoches_
                    # print('222', len(drop_idx_))



            else:
                idx_epoches_ = idx_epoches
                drop_idx_sample = random.sample(idx_epoches_, (max(idx_dic1.values())))

                idx_epoches_ = list(set(idx_epoches).difference(set(drop_idx_sample)))
                drop_idx_ = idx_epoches_

                # print('333',len(drop_idx_))

            idx_epoches_all.append(idx_epoches_)
            drop_idx_all.append(drop_idx_)

        idx_epoches_all = np.array(idx_epoches_all)
        drop_idx_all = np.array(drop_idx_all)
        train_edges_pos = np.array(train_edges_pos)
        train_edges_neg = np.array(train_edges_neg)

        y_train_train = np.concatenate((train_edges_pos, np.ones(np.shape(train_edges_pos)[0]).reshape(-1, 1)), axis=1)
        y_train_test = np.concatenate((train_edges_neg, np.zeros(np.shape(train_edges_neg)[0]).reshape(-1, 1)), axis=1)
        y_test_train = np.concatenate((test_edges_pos, np.ones(np.shape(test_edges_pos)[0]).reshape(-1, 1)), axis=1)
        y_test_test = np.concatenate((test_edges_neg, np.zeros(np.shape(test_edges_neg)[0]).reshape(-1, 1)), axis=1)

        print(np.shape(train_edges_pos), np.shape(idx_epoches_all), np.shape(drop_idx_all), np.shape(aug1s_embed))
        pos_train_edge_embs0 = get_edge_embeddings(train_edges_pos, aug1s_embed, idx_epoches_all)
        neg_train_edge_embs0 = get_edge_embeddings(train_edges_neg, aug1s_embed, drop_idx_all)

        pos_test_edge_embs0 = get_edge_embeddings(test_edges_pos, aug1s_embed, idx_epoches_all)
        neg_test_edge_embs0 = get_edge_embeddings(test_edges_neg, aug1s_embed, drop_idx_all)

        pos_train_edge_embs1 = get_edge_embeddings(train_edges_pos, aug2s_embed, idx_epoches_all)
        neg_train_edge_embs1 = get_edge_embeddings(train_edges_neg, aug2s_embed, drop_idx_all)

        pos_test_edge_embs1 = get_edge_embeddings(test_edges_pos, aug2s_embed, idx_epoches_all)
        neg_test_edge_embs1 = get_edge_embeddings(test_edges_neg, aug2s_embed, drop_idx_all)

        X_train = np.concatenate((pos_train_edge_embs0, neg_train_edge_embs0), axis=0)
        X_test = np.concatenate((pos_test_edge_embs0, neg_test_edge_embs0), axis=0)
        y_train = np.concatenate((y_train_train, y_train_test), axis=0)
        y_test = np.concatenate((y_test_train, y_test_test), axis=0)

        # # ######################################################################

        from sklearn import metrics
        from sklearn.neural_network import MLPClassifier

        mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,
                            max_iter=1000)

        mlp.fit(X_train, y_train[:, 2])

        print("Training set score: %f" % mlp.score(X_train, y_train[:, 2]))
        print("Test set score: %f" % mlp.score(X_test, y_test[:, 2]))

        y_score = mlp.predict(X_test)
        print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
        print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

        acc_mlp_sim_embed0 = accuracy_score(y_score, y_test[:, 2])

        tsts = []
        for i in range(len(y_score)):
            node1 = y_test[i][0]
            node2 = y_test[i][1]

            tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'node1', 'node2']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-embed-mlp_sim0.csv".format(res_dir, dt))

        # # ######################################################################

        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(max_depth=150, random_state=0)
        rf.fit(X_train, y_train[:, 2])

        print("Training set score: %f" % rf.score(X_train, y_train[:, 2]))
        print("Test set score: %f" % rf.score(X_test, y_test[:, 2]))

        y_score = rf.predict(X_test)
        print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
        print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

        acc_rf_sim_embed0 = accuracy_score(y_score, y_test[:, 2])

        tsts = []
        for i in range(len(y_score)):
            node1 = y_test[i][0]
            node2 = y_test[i][1]

            tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'node1', 'node2']

        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-embed-rf_sim0.csv".format(res_dir, dt))

        # # ######################################################################

        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC

        svm = OneVsRestClassifier(SVC())
        svm.fit(X_train, y_train[:, 2])

        print("Training set score: %f" % svm.score(X_train, y_train[:, 2]))
        print("Test set score: %f" % svm.score(X_test, y_test[:, 2]))

        y_score = svm.predict(X_test)
        print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
        print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

        acc_svm_sim_embed0 = accuracy_score(y_score, y_test[:, 2])

        tsts = []
        for i in range(len(y_score)):
            node1 = y_test[i][0]
            node2 = y_test[i][1]

            tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'node1', 'node2']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-embed-svm_sim0.csv".format(res_dir, dt))

        X_train = np.concatenate((pos_train_edge_embs1, neg_train_edge_embs1), axis=0)
        X_test = np.concatenate((pos_test_edge_embs1, neg_test_edge_embs1), axis=0)
        y_train = np.concatenate((y_train_train, y_train_test), axis=0)
        y_test = np.concatenate((y_test_train, y_test_test), axis=0)

        # # ######################################################################

        from sklearn import metrics
        from sklearn.neural_network import MLPClassifier

        mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,
                            max_iter=1000)

        mlp.fit(X_train, y_train[:, 2])

        print("Training set score: %f" % mlp.score(X_train, y_train[:, 2]))
        print("Test set score: %f" % mlp.score(X_test, y_test[:, 2]))

        y_score = mlp.predict(X_test)
        print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
        print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

        acc_mlp_sim_embed1 = accuracy_score(y_score, y_test[:, 2])

        tsts = []
        for i in range(len(y_score)):
            node1 = y_test[i][0]
            node2 = y_test[i][1]

            tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'node1', 'node2']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-embed-mlp_sim1.csv".format(res_dir, dt))

        # # ######################################################################

        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(max_depth=150, random_state=0)
        rf.fit(X_train, y_train[:, 2])

        print("Training set score: %f" % rf.score(X_train, y_train[:, 2]))
        print("Test set score: %f" % rf.score(X_test, y_test[:, 2]))

        y_score = rf.predict(X_test)
        print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
        print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

        acc_rf_sim_embed1 = accuracy_score(y_score, y_test[:, 2])

        tsts = []
        for i in range(len(y_score)):
            node1 = y_test[i][0]
            node2 = y_test[i][1]

            tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'node1', 'node2']

        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-embed-rf_sim1.csv".format(res_dir, dt))

        # # ######################################################################

        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC

        svm = OneVsRestClassifier(SVC())
        svm.fit(X_train, y_train[:, 2])

        print("Training set score: %f" % svm.score(X_train, y_train[:, 2]))
        print("Test set score: %f" % svm.score(X_test, y_test[:, 2]))

        y_score = svm.predict(X_test)
        print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
        print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

        acc_svm_sim_embed1 = accuracy_score(y_score, y_test[:, 2])

        tsts = []
        for i in range(len(y_score)):
            node1 = y_test[i][0]
            node2 = y_test[i][1]

            tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'node1', 'node2']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-embed-svm_sim1.csv".format(res_dir, dt))

        pos_train_edge_embs1 = np.concatenate((pos_train_edge_embs0, pos_train_edge_embs1), axis=1)
        neg_train_edge_embs1 = np.concatenate((neg_train_edge_embs0, neg_train_edge_embs1), axis=1)

        pos_test_edge_embs1 = np.concatenate((pos_test_edge_embs0, pos_test_edge_embs1), axis=1)
        neg_test_edge_embs1 = np.concatenate((neg_test_edge_embs0, neg_test_edge_embs1), axis=1)

        X_train = np.concatenate((pos_train_edge_embs1, neg_train_edge_embs1), axis=0)
        X_test = np.concatenate((pos_test_edge_embs1, neg_test_edge_embs1), axis=0)
        y_train = np.concatenate((y_train_train, y_train_test), axis=0)
        y_test = np.concatenate((y_test_train, y_test_test), axis=0)

        # # ######################################################################

        from sklearn import metrics
        from sklearn.neural_network import MLPClassifier

        mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,
                            max_iter=1000)

        mlp.fit(X_train, y_train[:, 2])

        print("Training set score: %f" % mlp.score(X_train, y_train[:, 2]))
        print("Test set score: %f" % mlp.score(X_test, y_test[:, 2]))

        y_score = mlp.predict(X_test)
        print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
        print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

        acc_mlp_sim_embed2 = accuracy_score(y_score, y_test[:, 2])

        tsts = []
        for i in range(len(y_score)):
            node1 = y_test[i][0]
            node2 = y_test[i][1]

            tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'node1', 'node2']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-embed-mlp_sim2.csv".format(res_dir, dt))

        # # ######################################################################

        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(max_depth=150, random_state=0)
        rf.fit(X_train, y_train[:, 2])

        print("Training set score: %f" % rf.score(X_train, y_train[:, 2]))
        print("Test set score: %f" % rf.score(X_test, y_test[:, 2]))

        y_score = rf.predict(X_test)
        print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
        print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

        acc_rf_sim_embed2 = accuracy_score(y_score, y_test[:, 2])

        tsts = []
        for i in range(len(y_score)):
            node1 = y_test[i][0]
            node2 = y_test[i][1]

            tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'node1', 'node2']

        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-embed-rf_sim2.csv".format(res_dir, dt))

        # # ######################################################################

        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC

        svm = OneVsRestClassifier(SVC())
        svm.fit(X_train, y_train[:, 2])

        print("Training set score: %f" % svm.score(X_train, y_train[:, 2]))
        print("Test set score: %f" % svm.score(X_test, y_test[:, 2]))

        y_score = svm.predict(X_test)
        print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
        print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

        acc_svm_sim_embed2 = accuracy_score(y_score, y_test[:, 2])

        tsts = []
        for i in range(len(y_score)):
            node1 = y_test[i][0]
            node2 = y_test[i][1]

            tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'node1', 'node2']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-embed-svm_sim2.csv".format(res_dir, dt))

        print(acc_mlp_sim_embed0, acc_rf_sim_embed0, acc_svm_sim_embed0)

        print(acc_mlp_sim_embed1, acc_rf_sim_embed1, acc_svm_sim_embed1)

        print(acc_mlp_sim_embed2, acc_rf_sim_embed2, acc_svm_sim_embed2)


        results.append(
            [acc_mlp_sim_embed0, acc_rf_sim_embed0, acc_svm_sim_embed0, acc_mlp_sim_embed1, acc_rf_sim_embed1,
             acc_svm_sim_embed1, acc_mlp_sim_embed2, acc_rf_sim_embed2, acc_svm_sim_embed2])

result_all = pd.DataFrame(data=results)
result_all.to_csv("{}/results_all-.csv".format(res_dir))
