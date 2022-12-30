import numpy
import argparse
import os.path as osp
import random
import nni
import numpy as np
import os
import networkx as nx
import pickle as pk

import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected
from sklearn.metrics import f1_score,accuracy_score
import itertools

from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights_mia, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset
from pGRACE import preprocessing
import pandas as pd


###python train-mia-2.py --device cuda:0 --dataset Cora --param local:cora.json --drop_scheme degree

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

        # if idx_epoches_all[i]!=[]:
        #     embs_1_cos.append(sim1[idx_epoches_all[i]])
        #     embs_1_dot.append(sim2[idx_epoches_all[i]])
        #
        # i+=1

        # print(embs_1_cos, embs_2_cos)

        # exit()

        if len(idx_epoches_all[i])>0:
            # print('aaa')
            embs_1_cos.append(sim1[idx_epoches_all[i]])
            embs_1_dot.append(sim2[idx_epoches_all[i]])

        i+=1

    # print(embs_1_cos, embs_2_cos)

    if np.shape(idx_epoches_all)[1]==1:
        embs = np.concatenate((np.array(list(itertools.chain.from_iterable(embs_1_cos))).reshape(-1,1), np.array(list(itertools.chain.from_iterable(embs_1_dot))).reshape(-1,1)), axis=1)
    else:

        embs = np.concatenate((np.array(embs_1_cos),np.array(embs_1_dot)),axis=1)

    return embs

def train():
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        global drop_weights

        if param['drop_scheme'] == 'uniform':
            return dropout_adj(edges_train_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(edges_train_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    # print(edge_index_1)
    # exit()
    x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])

    # print(x_1)
    # print(edge_index_1.size(),x_1.size())

    # exit()

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    z_train = model(data.x, edges_train_index)

    loss = model.loss(z1, z2, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)
    loss.backward()
    optimizer.step()

    return loss.item(),edge_index_1,edge_index_2,z_train, z1, z2


def test(final=False):
    model.eval()
    z = model(data.x, edges_train_index)

    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param', type=str, default='local:wikics.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 128,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'


    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)

    device = torch.device(args.device)

    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)

    dt=args.dataset
    ratio=param['drop_edge_rate_1']
    r=param[f'drop_feature_rate_1']

    data = dataset[0]
    data = data.to(device)

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
    g.add_nodes_from(list(range(np.shape(data.x.detach().cpu().numpy())[0])))
    g.add_edges_from(edge_index0_all)
    num_nodes=g.number_of_nodes()

    print('***',num_nodes,dt)
    # exit()
    adj_sparse = nx.to_scipy_sparse_matrix(g)
    random.seed(42)
    train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split
    # # print(adj_train)
    # #

    res_dir='%s-gca-mia-white-2'%(dt)

    with open('./%s/%s-train_test_split' % (res_dir, dt), 'wb') as f:
        pk.dump(train_test_split, f)

    f2 = open('./%s/%s-train_test_split' % (res_dir, dt), 'rb')
    train_test_split = pk.load(f2, encoding='latin1')

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split
    print('***',num_nodes,dt)
    g_train0 = nx.Graph()
    g_train0.add_nodes_from(list(range(num_nodes)))
    g_train0.add_edges_from(train_edges)

    print('***', g_train0.number_of_nodes())

    # g_train0 = nx.from_scipy_sparse_matrix(
    #     adj_train)  # new graph object with only non-hidden edges, keep all the original nodes

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

    loop_nodes=np.arange(0,g.number_of_nodes())
    train_edges_3=np.concatenate((loop_nodes.reshape(-1,1),loop_nodes.reshape(-1,1)),axis=1)
    train_edges_3 = np.transpose(np.array(train_edges_3))
    edges_train_index=np.concatenate((train_edges_1,train_edges_2,train_edges_3),axis=1)

    # edges_train_index = np.concatenate((train_edges_1, train_edges_2), axis=1)

    edges_train_index = torch.from_numpy(np.array(edges_train_index)).long().to(device)

    # generate split
    split = generate_split(num_nodes, train_ratio=0.1, val_ratio=0.1)

    if args.save_split:
        torch.save(split, args.save_split)
    elif args.load_split:
        split = torch.load(args.load_split)

    encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(edges_train_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(edges_train_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights_mia(g_train0,edges_train_index,data.x,num_nodes).to(device)
    else:
        drop_weights = None

    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(edges_train_index)
        node_deg = degree(edge_index_[1])
        # print(node_deg)
        print(node_deg.size())
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            print(data.x)
            print(data.x.size())
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
            # exit()
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(edges_train_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    log = args.verbose.split(',')


    losss1=[]
    losss2=[]

    z1_trains=[]
    z2_trains=[]

    edge_index1_trains=[]
    edge_index2_trains = []


    best_valid_loss=99999999

    patience = 50


    for epoch in range(1, param['num_epochs'] + 1):
        loss,edge_index1,edge_index2,z_train, z1_train, z2_train = train()

        if loss < best_valid_loss:
            best_valid_loss = loss
            trail_count = 0
            best_epoch = epoch
            torch.save(encoder.state_dict(), os.path.join('./checkpoint',
                                                                'tmp',
                                                                f'gca_{dt}_{ratio}_{r}_best.pt'))

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

        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        if epoch % 100 == 0:
            acc = test()

            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')

    encoder.load_state_dict(torch.load(os.path.join('./checkpoint', 'tmp',
                                                          f'gca_{dt}_{ratio}_{r}_best.pt')))

    # edge_index1_trains=np.array(edge_index1_trains)
    # edge_index2_trains=np.array(edge_index2_trains)

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


    acc = test(final=True)

    if 'final' in log:
        print(f'{acc}')


    aug1s=edge_index1_trains
    aug2s=edge_index2_trains
    aug1s_embed=z1_trains_
    aug2s_embed=z2_trains_
    edges_train_all=train_edges0
    edges_test_all=edges_test0

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
    # print()

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
    # print('111',drop1s_pos_idx)

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

                    if epoches-max(idx_dic1.values()) - len(drop_idx)>0:
                        drop_idx_sample2 = random.sample(idx_epoches_,
                                                         (epoches - max(idx_dic1.values()) - len(drop_idx)))
                        drop_idx_sample = random.sample(idx_epoches_, (max(idx_dic1.values()) - len(drop_idx)))
                        idx_epoches_ = list(set(idx_epoches_).difference(set(drop_idx_sample)))

                        drop_idx_ = list(drop_idx) + drop_idx_sample2

                        # print('000', len(drop_idx_),len(idx_epoches_))

                    else:

                        drop_idx_sample = random.sample(idx_epoches_, (max(idx_dic1.values()) - len(drop_idx)))
                        idx_epoches_ = list(set(idx_epoches_).difference(set(drop_idx_sample)))
                        if len(drop_idx)>len(idx_epoches_):
                            drop_idx_sample2 = list(random.sample(drop_idx,len(idx_epoches_)))
                            drop_idx_ = drop_idx_sample2
                        else:
                            drop_idx_sample2 = list(random.sample(drop_idx_sample, (len(idx_epoches_)-len(list(drop_idx)))))
                            drop_idx_ = list(drop_idx) +drop_idx_sample2

                        print(len(drop_idx))



                        # print('111', len(drop_idx_),len(idx_epoches_))

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


        # exit()

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
        print('00')
        print(drop_idx_all)
        neg_train_edge_embs0 = get_edge_embeddings2(train_edges_neg, aug1s_embed, drop_idx_all)
        print('11')

        pos_test_edge_embs0 = get_edge_embeddings2(test_edges_pos, aug1s_embed, idx_epoches_all)
        print('22')
        neg_test_edge_embs0 = get_edge_embeddings2(test_edges_neg, aug1s_embed, drop_idx_all)
        print('33')

        pos_train_edge_embs1 = get_edge_embeddings2(train_edges_pos, aug2s_embed, idx_epoches_all)
        neg_train_edge_embs1 = get_edge_embeddings2(train_edges_neg, aug2s_embed, drop_idx_all)

        pos_test_edge_embs1 = get_edge_embeddings2(test_edges_pos, aug2s_embed, idx_epoches_all)
        neg_test_edge_embs1 = get_edge_embeddings2(test_edges_neg, aug2s_embed, drop_idx_all)

        X_train = np.concatenate((pos_train_edge_embs0, neg_train_edge_embs0), axis=0)
        X_test = np.concatenate((pos_test_edge_embs0, neg_test_edge_embs0), axis=0)
        y_train = np.concatenate((y_train_train, y_train_test), axis=0)
        y_test = np.concatenate((y_test_train, y_test_test), axis=0)


        tuples_ = (pos_train_edge_embs0, neg_train_edge_embs0, pos_test_edge_embs0, neg_test_edge_embs0)

        with open('./%s/%s-edge-sim0' % (res_dir, dt), 'wb') as f:
            pk.dump(tuples_, f)

        tuples = (X_train, X_test, y_train, y_test)

        with open('./%s/%s-train-test-data0' % (res_dir, dt), 'wb') as f:
            pk.dump(tuples, f)


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


        tuples_ = (pos_train_edge_embs1, neg_train_edge_embs1, pos_test_edge_embs1, neg_test_edge_embs1)

        with open('./%s/%s-edge-sim1' % (res_dir, dt), 'wb') as f:
            pk.dump(tuples_, f)

        tuples = (X_train, X_test, y_train, y_test)

        with open('./%s/%s-train-test-data1' % (res_dir, dt), 'wb') as f:
            pk.dump(tuples, f)


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
result_all.to_csv("{}/results_all-%s-%s.csv".format(res_dir,ratio,r))

