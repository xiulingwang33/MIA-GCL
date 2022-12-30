import torch
import os.path as osp
import GCL.losses as L

import sys
import os

print(sys.path)

sys.path.append('./GCL2')


import torch_geometric.transforms as T

#
import GCL2.augmentors as A

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL2.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid,Amazon

import numpy as np
import random
import pandas as pd

from GCL2 import preprocessing

import networkx as nx
import pickle as pk
import itertools
from sklearn.metrics import f1_score,accuracy_score


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



class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z1 = self.encoder1(x1, edge_index1, edge_weight1)
        z2 = self.encoder2(x2, edge_index2, edge_weight2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1(*self.corruption(x1, edge_index1, edge_weight1))
        z2n = self.encoder2(*self.corruption(x2, edge_index2, edge_weight2))
        return z1, z2, g1, g2, z1n, z2n, edge_index1, edge_index2


def train(dt,device,encoder_model, contrast_model, data,edges_train_index, optimizer):



    encoder_model.train()
    optimizer.zero_grad()


    z1, z2, g1, g2, z1n, z2n, edge_index1, edge_index2 = encoder_model(data.x, edges_train_index)

    loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item(),z1, z2,edge_index1, edge_index2


def test(encoder_model, data,edges_train_index):
    encoder_model.eval()
    z1, z2, _, _, _, _,_,_ = encoder_model(data.x,edges_train_index)
    z = z1 + z2
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result,z1, z2,z


def main():
    device = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # device = torch.device(f'cuda:{os.environ["CUDA_VISIBLE_DEVICES"]}')
    # torch.cuda.set_device(4)
    # device='cpu'

    path = osp.join(osp.expanduser('~'), 'datasets')
    dt='cora'  ##cora , citeseer
    dataset = Planetoid(path, name=dt, transform=T.NormalizeFeatures())

    dataset = Amazon(path, dt, pre_transform=None)  # #Computers, photo

    data = dataset[0].to(device)

    aug1 = A.Identity()
    alpha = 0.2
    aug2 = A.PPRDiffusion(alpha)
    gconv1 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2).to(device)
    gconv2 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2).to(device)
    encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=512).to(device)
    contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.001)

    edge_index0 = data.edge_index.detach().cpu().numpy()
    edge_index0=edge_index0.transpose()
    edges0_idx=edge_index0[:,0]*np.shape(data.x)[0]+edge_index0[:,1]
    print(edges0_idx)




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


    ratio=alpha

    res_dir = '%s-mvgrl-mia-white-2-%s' % (dt, alpha)


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

    test_edges0=edges_test0

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
    test_edges_1 = np.concatenate((test_edges0[:, 1].reshape(-1, 1), test_edges0[:, 0].reshape(-1, 1)), axis=1)
    test_edges_1 = np.transpose(np.array(test_edges_1))
    test_edges_2 = np.transpose(np.array(test_edges0))


    edges_train_index=np.concatenate((train_edges_1,train_edges_2),axis=1)

    edges_test_index = np.concatenate((test_edges_1, test_edges_2), axis=1)


    edges_train_index = torch.from_numpy(np.array(edges_train_index)).long().to(device)
    edges_test_index = torch.from_numpy(np.array(edges_test_index)).long().to(device)



    z1_trains = []
    z2_trains = []

    edge_index1_trains=[]
    edge_index2_trains = []

    best_valid_loss = 99999999


    with tqdm(total=1000, desc='(T)') as pbar:
        for epoch in range(1, 1001):
            loss,z1, z2,edge_index1, edge_index2 = train(dt,device,encoder_model, contrast_model, data,edges_train_index, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

            patience = 100

            if loss < best_valid_loss:
                best_valid_loss = loss
                trail_count = 0
                best_epoch = epoch
                torch.save(encoder_model.state_dict(), os.path.join('./checkpoint',
                                                                    'tmp',
                                                                    f'mvgrl_{dt}_{ratio}_best.pt'))
                z1_trains.append(z1.detach().cpu().numpy())
                z2_trains.append(z2.detach().cpu().numpy())

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

                    z1_trains.append(z1.detach().cpu().numpy())
                    z2_trains.append(z2.detach().cpu().numpy())

        encoder_model.load_state_dict(torch.load(os.path.join('./checkpoint', 'tmp',
                                                              f'mvgrl_{dt}_{ratio}_best.pt')))

    test_result,z1, z2,z = test(encoder_model, data,edges_train_index)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    train_preds = test_result["output_train"]

    # res_dir = '%s-mvgrl-node-mia-mi' % (dt)

    emb_matrix0 = z.detach().cpu().numpy()

    emb_matrix1 = z1.detach().cpu().numpy()
    emb_matrix2 = z2.detach().cpu().numpy()

    with open('./%s/%s-embed0.txt' % (res_dir, dt), 'w') as f:
        f.write('%d %d\n' % (np.shape(emb_matrix0)[0], np.shape(emb_matrix0)[1]))
        for item in emb_matrix0:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    with open('./%s/%s-output_train.txt' % (res_dir, dt), 'w') as f:
        for item in train_preds:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()


    edge_index1 = edge_index1.detach().cpu().numpy()
    edge_index2 = edge_index2.detach().cpu().numpy()

    # print(edge_index0, edge_index1, edge_index2, edges_test)

    print(np.shape(edge_index0), np.shape(edge_index1), np.shape(edge_index2))


    #
    # for name, param in encoder_model.named_parameters():
    #     print(name,param.size())
    #
    # for name, param in contrast_model.named_parameters():
    #     print('***',name,param.size())

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


    return emb_matrix0,train_preds,train_edges0,edges_test0,data.y,res_dir,z1_trains,z2_trains,dt,edge_index1_trains,edge_index2_trains,z1_trains_,z2_trains_



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
    emb_matrix, output_train, edges_train_all, edges_test_all, label, res_dir,emb_matrix1,emb_matrix2,dt,aug1s,aug2s,aug1s_embed,aug2s_embed =main()

    edges_train_all = np.array(edges_train_all)
    edges_test_all = np.array(edges_test_all)

    train_preds = output_train

    train_range1 = list(np.arange(np.shape(edges_train_all)[0]))
    # train_range2 = list(np.arange(np.shape(edges_train_inter)[0]))

    # Train-set edge embeddings
    train_preds_sampled_idx1 = np.array(random.sample(train_range1, np.shape(edges_test_all)[0]))
    # train_preds_sampled_idx2 = np.array(random.sample(train_range2, np.shape(edges_test_all)[0]))

    print(train_preds_sampled_idx1)

    # train_preds_sampled1 = np.array(edges_train_all)[train_preds_sampled_idx1]
    train_edges_sampled1 = np.array(edges_train_all)[train_preds_sampled_idx1, :]

    # train_preds_sampled2 = np.array(edges_train_all)[train_preds_sampled_idx2]
    # train_edges_sampled2 = np.array(edges_train_inter)[train_preds_sampled_idx2,:]

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

    print(train_edges_sampled1)
    print(edges_test_all)

    z1_trains = emb_matrix1
    z2_trains = emb_matrix2

    ylabel = [1] * len(train_preds_sampled_idx1) + [0] * len(train_preds_sampled_idx1)

    from sklearn.model_selection import train_test_split

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
    add1s_pos_idx = []
    add2s_pos_idx = []

    for aug_idx in aug1s_idx:
        drop_idx = np.intersect1d(train_edges_neg_idx, aug_idx)
        add1s_pos_idx.append(drop_idx)

    for aug_idx in aug2s_idx:
        drop_idx = np.intersect1d(train_edges_neg_idx, aug_idx)
        add2s_pos_idx.append(drop_idx)

    # print(drop1s_pos_idx)
    # print(drop2s_pos_idx)


    with open('./%s/%s-add1s_pos_idx.txt' % (res_dir, dt), 'w') as f:
        for item in add1s_pos_idx:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    with open('./%s/%s-add2s_pos_idx.txt' % (res_dir, dt), 'w') as f:
        for item in add2s_pos_idx:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    # file_name='./%s/%s-add1s_pos_idx.txt' % (res_dir,dt)
    # add1s_pos_idx0=readedges2(file_name)
    # # print(drop1s_pos_idx)

    file_name = './%s/%s-add2s_pos_idx.txt' % (res_dir, dt)
    add2s_pos_idx0 = readedges2(file_name)

    print('####', add2s_pos_idx0[0])

    # print(drop2s_pos_idx0[0])

    # print(drop2s_pos_idx0[0])

    iterations = np.shape(add2s_pos_idx0)[0]

    # iter_ratios=[0.2,0.4,0.6,0.8,1]
    iter_ratios = [1]

    # drop1s_pos_idx0=add1s_pos_idx0
    drop2s_pos_idx0 = add2s_pos_idx0

    # results=[]
    for iters in iter_ratios:
        iter_ = int(iterations * iters) - 1

        # drop1s_pos_idx=drop1s_pos_idx0[0:iter_]
        drop2s_pos_idx = drop2s_pos_idx0[0:iter_]

        # drop1s_pos_idx_=list(itertools.chain.from_iterable(drop1s_pos_idx))
        drop2s_pos_idx_ = list(itertools.chain.from_iterable(drop2s_pos_idx))

        # print(len(drop1s_pos_idx_),len(drop2s_pos_idx_))
        # set1=list(set(drop1s_pos_idx_))
        set2 = list(set(drop2s_pos_idx_))
        print(len(set2))
        set0 = list(set(set2))
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

        # i=0
        # for idx in drop1s_pos_idx:
        #     for j in idx:
        #         idx_dic1[j]+=1
        #         idx_dic1_[j].append(i)
        #     i+=1

        i = 0
        for idx in drop2s_pos_idx:
            for j in idx:
                idx_dic2[j] += 1
                idx_dic2_[j].append(i)
            i += 1

        # print(min(idx_dic1.values()),max(idx_dic1.values()))
        print(min(idx_dic2.values()), max(idx_dic2.values()))

        # print(idx_dic1,idx_dic2)
        # idx_dic0=[]
        # for idx in set0:
        #     idx_dic0.append(idx_dic1[idx]+idx_dic2[idx])
        # # print(idx_dic0)

        idx_dic0 = idx_dic2

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
        for i in train_edges_neg_idx:

            if i in idx_dic2_.keys():

                drop_idx = idx_dic2_[i]
                idx_epoches_ = list(set(idx_epoches).difference(set(drop_idx)))
                if len(drop_idx) < max(idx_dic2.values()):
                    drop_idx_sample2 = random.sample(idx_epoches_, (epoches - max(idx_dic2.values()) - len(drop_idx)))
                    drop_idx_sample = random.sample(idx_epoches_, (max(idx_dic2.values()) - len(drop_idx)))
                    idx_epoches_ = list(set(idx_epoches_).difference(set(drop_idx_sample)))
                    drop_idx_ = list(drop_idx) + drop_idx_sample2
                else:
                    idx_epoches_ = list(set(idx_epoches_))
                    drop_idx_ = idx_epoches_
                    drop_idx_ = idx_epoches_

            else:
                idx_epoches_ = idx_epoches
                drop_idx_sample = random.sample(idx_epoches_, (max(idx_dic2.values())))

                idx_epoches_ = list(set(idx_epoches).difference(set(drop_idx_sample)))
                drop_idx_ = idx_epoches_

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

        print(np.shape(train_edges_pos), np.shape(idx_epoches_all), np.shape(aug1s_embed))
        pos_train_edge_embs0 = get_edge_embeddings(train_edges_pos, drop_idx_all)
        neg_train_edge_embs0 = get_edge_embeddings(train_edges_neg, aug1s_embed, idx_epoches_all)

        pos_test_edge_embs0 = get_edge_embeddings(test_edges_pos, aug1s_embed, drop_idx_all)
        neg_test_edge_embs0 = get_edge_embeddings(test_edges_neg, aug1s_embed, idx_epoches_all)

        pos_train_edge_embs1 = get_edge_embeddings(train_edges_pos, aug2s_embed, drop_idx_all)
        neg_train_edge_embs1 = get_edge_embeddings(train_edges_neg, aug2s_embed, idx_epoches_all)

        pos_test_edge_embs1 = get_edge_embeddings(test_edges_pos, aug2s_embed, drop_idx_all)
        neg_test_edge_embs1 = get_edge_embeddings(test_edges_neg, aug2s_embed, idx_epoches_all)

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
result_all.to_csv("{}/results_all.csv".format(res_dir))

