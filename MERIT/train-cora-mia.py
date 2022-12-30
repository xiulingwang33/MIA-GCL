# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import random
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
from utils import process
from utils import aug
from modules.gcn import GCNLayer
from net.merit import MERIT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

import networkx as nx
import pickle as pk
import itertools


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--data', type=str, default='citeseer')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--eval_every', type=int, default=100)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--sample_size', type=int, default=2000)
parser.add_argument('--patience', type=int, default=25)
parser.add_argument('--sparse', type=str_to_bool, default=True)

parser.add_argument('--input_dim', type=int, default=1433)
parser.add_argument('--gnn_dim', type=int, default=128)
parser.add_argument('--proj_dim', type=int, default=128)
parser.add_argument('--proj_hid', type=int, default=128)
parser.add_argument('--pred_dim', type=int, default=128)
parser.add_argument('--pred_hid', type=int, default=128)
parser.add_argument('--momentum', type=float, default=0.8)
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--drop_edge', type=float, default=0.4)
parser.add_argument('--drop_feat1', type=float, default=0.4)
parser.add_argument('--drop_feat2', type=float, default=0.4)

args = parser.parse_args()
torch.set_num_threads(4)

if args.data=='cora':
    args.input_dim=1433
elif args.data=='citeseer':
    args.input_dim = 3703


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

def get_edge_embeddings2(edge_list, emb_matrixs,idx_epoches_all):
    embs = []
    embs_1_cos=[]
    embs_1_dot=[]
    embs_2_cos = []
    embs_2_dot = []

    i=0
    for edge in edge_list:
        node1 = int(edge[0])
        node2 = int(edge[1])
        emb_1_cos=[]
        emb_1_dot = []
        # print(i)
        # print(idx_epoches_all[i,:])
        # print(len(idx_epoches_all[i,:]))

        emb1 = emb_matrixs[:,node1,:]
        emb2 = emb_matrixs[:, node2, :]


        # print(emb1)
        # print(np.shape(emb1))

        edge_emb = np.multiply(emb1, emb2)

        # print(edge_emb)
        # print(np.shape(edge_emb))

        sim2=np.sum(edge_emb,axis=1)

        tmp1=np.multiply(emb1, emb1)
        tmp2 = np.multiply(emb2, emb2)
        # print(np.shape(tmp1))

        tmp1 = np.sum(tmp1, axis=1)
        tmp2 = np.sum(tmp2, axis=1)

        # print('mmmm',np.shape(tmp2))

        tmp=np.multiply(tmp1,tmp2)

        # print(np.shape(tmp))

        sim1=[]
        for s in range(len(sim2)):
            sim1.append(sim2[s]/(np.sqrt(tmp[s])))

        sim1=np.array(sim1)

        # print(sim1)
        # print(np.shape(sim1))
        # print(np.shape(sim2))

        if idx_epoches_all[i]!=[]:
            embs_1_cos.append(sim1[idx_epoches_all[i]])
            embs_1_dot.append(sim2[idx_epoches_all[i]])

        i+=1

        # print(embs_1_cos, embs_2_cos)

        # exit()

    embs = np.concatenate((np.array(embs_1_cos),np.array(embs_1_dot)),axis=1)

    return embs

    # print(np.shape(embs_1_cos))

    # embs_1_cos = np.array(embs_1_cos)
    # embs_2_cos = np.array(embs_2_cos)
    # embs_1_dot = np.array(embs_1_dot)
    # embs_2_dot = np.array(embs_2_dot)
    # # print(embs_1_cos,embs_2_cos,embs_1_dot,embs_2_dot)
    #
    # print(np.shape(embs_1_cos),np.shape(embs_2_cos),np.shape(embs_1_dot),np.shape(embs_2_dot ))
    #
    # # exit()
    # return embs_1_cos,embs_2_cos,embs_1_dot,embs_2_dot




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


#

#
# def evaluation(adj, feat, gnn, idx_train, idx_test, sparse):
#     clf = LogisticRegression(random_state=0, max_iter=2000)
#     model = GCNLayer(input_size, gnn_output_size)  # 1-layer
#     model.load_state_dict(gnn.state_dict())
#     with torch.no_grad():
#         embeds1 = model(feat, adj, sparse)
#         embeds2 = model(feat, diff, sparse)
#         train_embs = embeds1[0, idx_train] + embeds2[0, idx_train]
#         test_embs = embeds1[0, idx_test] + embeds2[0, idx_test]
#         train_labels = torch.argmax(labels[0, idx_train], dim=1)
#         test_labels = torch.argmax(labels[0, idx_test], dim=1)
#     embed=embeds1+embeds2
#     clf.fit(train_embs, train_labels)
#     pred_test_labels = clf.predict(test_embs)
#     pred_all = clf.predict(embed)
#     return accuracy_score(test_labels, pred_test_labels),embed,pred_all

def evaluation(adj, feat, gnn, idx_train, idx_test, sparse):
    clf = LogisticRegression(random_state=0, max_iter=2000)
    model = GCNLayer(input_size, gnn_output_size)  # 1-layer
    model.load_state_dict(gnn.state_dict())
    with torch.no_grad():
        embeds1 = model(feat, adj, sparse)[0,:,:]
        # embeds2 = model(feat, diff, sparse)
        train_embs = embeds1[idx_train,:]
        test_embs = embeds1[idx_test,:]
        train_labels = torch.argmax(labels[0, idx_train], dim=1)
        test_labels = torch.argmax(labels[0, idx_test], dim=1)
    embed=embeds1
    clf.fit(train_embs, train_labels)
    pred_test_labels = clf.predict(test_embs)
    pred_all = clf.predict(embed)
    return accuracy_score(test_labels, pred_test_labels),embed,pred_all

def evaluation2(adj, feat, gnn, sparse):
    # clf = LogisticRegression(random_state=0, max_iter=2000)
    model = GCNLayer(input_size, gnn_output_size)  # 1-layer
    model.load_state_dict(gnn.state_dict())
    with torch.no_grad():
        embeds1 = model(feat, adj, sparse)[0,:,:]
        # print(embeds1)
        # print(np.shape(embeds1))
    return embeds1



if __name__ == '__main__':

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # device = torch.device(f'cuda:{os.environ["CUDA_VISIBLE_DEVICES"]}')
    # torch.cuda.set_device(4)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    n_runs = args.runs
    eval_every_epoch = args.eval_every

    dataset = args.data
    input_size = args.input_dim

    gnn_output_size = args.gnn_dim
    projection_size = args.proj_dim
    projection_hidden_size = args.proj_hid
    prediction_size = args.pred_dim
    prediction_hidden_size = args.pred_hid
    momentum = args.momentum
    beta = args.beta
    alpha = args.alpha

    drop_edge_rate_1 = args.drop_edge
    drop_feature_rate_1 = args.drop_feat1
    drop_feature_rate_2 = args.drop_feat2

    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    sample_size = args.sample_size
    batch_size = args.batch_size
    patience = args.patience

    sparse = args.sparse

    # Loading dataset
    res_dir = '%s-merit-mia-white-%s-%s' % (dataset,drop_edge_rate_1 ,drop_feature_rate_1)

    adj, features, labels, idx_train, idx_val, idx_test,train_edges0,edges_test0= process.load_data_mia2(dataset,res_dir)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print('@@@',np.shape(edges_test0))

    dt=dataset
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

    # exit()
    g_train0 = nx.from_scipy_sparse_matrix(adj)

    number_of_nodes=g_train0.number_of_nodes()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    if os.path.exists('data/diff_{}_{}.npy'.format(dataset, alpha)):
        diff = np.load('data/diff_{}_{}.npy'.format(dataset, alpha), allow_pickle=True)
    else:
        diff = aug.gdc(adj, alpha=alpha, eps=0.0001)
        np.save('data/diff_{}_{}'.format(dataset, alpha), diff)

    features, _ = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    features = torch.FloatTensor(features[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])

    norm_adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    norm_diff = sp.csr_matrix(diff)
    if sparse:
        eval_adj = process.sparse_mx_to_torch_sparse_tensor(norm_adj)
        eval_diff = process.sparse_mx_to_torch_sparse_tensor(norm_diff)
    else:
        eval_adj = (norm_adj + sp.eye(norm_adj.shape[0])).todense()
        eval_diff = (norm_diff + sp.eye(norm_diff.shape[0])).todense()
        eval_adj = torch.FloatTensor(eval_adj[np.newaxis])
        eval_diff = torch.FloatTensor(eval_diff[np.newaxis])

    result_over_runs = []
    
    # Initiate models
    model = GCNLayer(input_size, gnn_output_size)
    merit = MERIT(gnn=model,
                  feat_size=input_size,
                  projection_size=projection_size,
                  projection_hidden_size=projection_hidden_size,
                  prediction_size=prediction_size,
                  prediction_hidden_size=prediction_hidden_size,
                  moving_average_decay=momentum, beta=beta).to(device)

    opt = torch.optim.Adam(merit.parameters(), lr=lr, weight_decay=weight_decay)

    results = []

    # Training
    best = 0
    patience_count = 0

    z1_trains=[]
    z2_trains=[]
    edge_index1_trains=[]
    edge_index2_trains = []
    for epoch in range(epochs):
        for _ in range(batch_size):
            # idx = np.random.randint(0, adj.shape[-1] - sample_size + 1)
            ba = adj
            bd = diff
            bd = sp.csr_matrix(np.matrix(bd))
            features = features.squeeze(0)
            bf = features

            aug_adj1 = aug.aug_random_edge(ba, drop_percent=drop_edge_rate_1)
            aug_adj2 = aug.aug_random_edge(ba, drop_percent=drop_edge_rate_1)
            aug_features1 = aug.aug_feature_dropout(bf, drop_percent=drop_feature_rate_1)
            aug_features2 = aug.aug_feature_dropout(bf, drop_percent=drop_feature_rate_2)

            aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
            aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

            if sparse:
                adj_1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1).to(device)
                adj_2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2).to(device)
            else:
                aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
                aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()
                adj_1 = torch.FloatTensor(aug_adj1[np.newaxis]).to(device)
                adj_2 = torch.FloatTensor(aug_adj2[np.newaxis]).to(device)

            print('***',adj_1.size(),adj_2.size())

            aug_features1 = aug_features1.to(device)
            aug_features2 = aug_features2.to(device)

            opt.zero_grad()
            loss = merit(adj_1, adj_2, aug_features1, aug_features2, sparse)
            loss.backward()
            opt.step()
            merit.update_ma()

            print(loss)

        edge_index1=[]
        edge_index2=[]
        g1=nx.Graph()
        g_1 = nx.from_scipy_sparse_matrix(aug_adj1)
        for u, v in g_1.edges():
            edge_index1.append([u,v])

        g2 = nx.Graph()
        g_2 = nx.from_scipy_sparse_matrix(aug_adj2)
        for u, v in g_2.edges():
            edge_index2.append([u,v])


        edge_index1_trains.append(np.array(edge_index1).T)
        edge_index2_trains.append(np.array(edge_index2).T)


        # print('11')

        z1_train= evaluation2(adj_1.cpu() , features, model, sparse)
        z2_train= evaluation2(adj_2.cpu(), features, model, sparse)

        # print('22')

        z1_trains.append(z1_train.detach().cpu().numpy())
        z2_trains.append(z2_train.detach().cpu().numpy())



        if epoch % eval_every_epoch == 0:
            acc,embs,pred_all = evaluation(eval_adj, features, model, idx_train, idx_test, sparse)
            if acc > best:
                best = acc
                patience_count = 0
            else:
                patience_count += 1
            results.append(acc)
            print('\t epoch {:03d} | loss {:.5f} | clf test acc {:.5f}'.format(epoch, loss.item(), acc))
            if patience_count >= patience:
                print('Early Stopping.')
                break
            
    result_over_runs.append(max(results))
    print('\t best acc {:.5f}'.format(max(results)))

    edges_train_all = train_edges0

    # emb_matrix0 = z.detach().cpu().numpy()
    # emb_matrix1=z1.detach().cpu().numpy()
    # emb_matrix2 = z2.detach().cpu().numpy()

    # edge_index1_trains=np.array(edge_index1_trains)
    # edge_index2_trains=np.array(edge_index2_trains)

    z1_trains_ = np.array(z1_trains)
    z2_trains_ = np.array(z2_trains)

    with open('./%s/%s-aug1.pkl' % (res_dir, dt), 'wb') as f:
        pk.dump(edge_index1_trains, f)

    with open('./%s/%s-aug2.pkl' % (res_dir, dt), 'wb') as f:
        pk.dump(edge_index2_trains, f)

    with open('./%s/%s-aug1-embed.pkl' % (res_dir, dt), 'wb') as f:
        pk.dump(z1_trains_, f)

    with open('./%s/%s-aug2-embed.pkl' % (res_dir, dt), 'wb') as f:
        pk.dump(z2_trains_, f)

    aug1s = edge_index1_trains
    aug2s = edge_index2_trains
    aug1s_embed = z1_trains_
    aug2s_embed = z2_trains_
    edges_train_all = train_edges0
    edges_test_all = edges_test0

    edges_train_all = np.array(edges_train_all)
    edges_test_all = np.array(edges_test_all)

    # train_preds = output_train

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

    # print(train_edges_sampled1)
    # print(edges_test_all)

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

    print(len(train_edges_sampled_idx), len(test_edges_sampled_idx), len(train_edges_pos_idx),
          len(train_edges_neg_idx))
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

    # print('####',drop1s_pos_idx0[0])

    # print(drop2s_pos_idx0[0])

    # print(drop2s_pos_idx0[0])


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
                    drop_idx_sample2 = random.sample(idx_epoches_,
                                                     (epoches - max(idx_dic1.values()) - len(drop_idx)))
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

        y_train_train = np.concatenate((train_edges_pos, np.ones(np.shape(train_edges_pos)[0]).reshape(-1, 1)),
                                       axis=1)
        y_train_test = np.concatenate((train_edges_neg, np.zeros(np.shape(train_edges_neg)[0]).reshape(-1, 1)),
                                      axis=1)
        y_test_train = np.concatenate((test_edges_pos, np.ones(np.shape(test_edges_pos)[0]).reshape(-1, 1)), axis=1)
        y_test_test = np.concatenate((test_edges_neg, np.zeros(np.shape(test_edges_neg)[0]).reshape(-1, 1)), axis=1)

        print(np.shape(train_edges_pos), np.shape(idx_epoches_all), np.shape(drop_idx_all), np.shape(aug1s_embed))
        pos_train_edge_embs0 = get_edge_embeddings2(train_edges_pos, aug1s_embed, idx_epoches_all)
        neg_train_edge_embs0 = get_edge_embeddings2(train_edges_neg, aug1s_embed, drop_idx_all)

        pos_test_edge_embs0 = get_edge_embeddings2(test_edges_pos, aug1s_embed, idx_epoches_all)
        neg_test_edge_embs0 = get_edge_embeddings2(test_edges_neg, aug1s_embed, drop_idx_all)

        pos_train_edge_embs1 = get_edge_embeddings2(train_edges_pos, aug2s_embed, idx_epoches_all)
        neg_train_edge_embs1 = get_edge_embeddings2(train_edges_neg, aug2s_embed, drop_idx_all)

        pos_test_edge_embs1 = get_edge_embeddings2(test_edges_pos, aug2s_embed, idx_epoches_all)
        neg_test_edge_embs1 = get_edge_embeddings2(test_edges_neg, aug2s_embed, drop_idx_all)

        X_train = np.concatenate((pos_train_edge_embs0, neg_train_edge_embs0), axis=0)
        X_test = np.concatenate((pos_test_edge_embs0, neg_test_edge_embs0), axis=0)
        y_train = np.concatenate((y_train_train, y_train_test), axis=0)
        y_test = np.concatenate((y_test_train, y_test_test), axis=0)

        print('MIA')

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
result_all.to_csv("{}/results_all-%s-%s.csv".format(res_dir, drop_edge_rate_1, drop_feature_rate_1))

