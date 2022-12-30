import argparse

from model import CCA_SSG_white, LogReg
from aug import random_aug_white
from dataset import load_mia_white2

import numpy as np
import torch as th
import torch.nn as nn

import warnings

warnings.filterwarnings('ignore')

import random
import pandas as pd
# import preprocessing
# import networkx as nx
import os

import pickle as pk
import itertools
from sklearn.metrics import f1_score,accuracy_score

def readedges2(file_name):
    file = open(file_name)

    dataMat = []
    for line in file.readlines():
        curLine = line.strip().split('\t')
        print(curLine)
        floatLine = list(map(int, curLine))
        # print(floatLine)
        dataMat.append(floatLine)

    # embeddings = np.array(dataMat,dtype='int')

    return dataMat

def get_edge_embeddings(edge_list, emb_matrixs,idx_epoches_all ):
    embs = []
    i=0
    print(',,,',np.shape(idx_epoches_all))

    for edge in edge_list:
        node1 = int(edge[0])
        node2 = int(edge[1])
        emb=[]
        # print(i)
        # print(idx_epoches_all[i,:])
        # print(len(idx_epoches_all[i,:]))
        # print(emb_matrixs[idx_epoches_all[i],:,:])
        for emb_matrix in emb_matrixs[idx_epoches_all[i],:,:]:
            emb1 = emb_matrix[node1]
            #print(np.shape(emb1))
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            sim1 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)+0.000000000000001)

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











parser = argparse.ArgumentParser(description='CCA-SSG')

parser.add_argument('--dataname', type=str, default='comp', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=3, help='GPU index.')
parser.add_argument('--epochs', type=int, default=1000, help='Training epochs.')
parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate of CCA-SSG.')
parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=0, help='Weight decay of CCA-SSG.')
parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of linear evaluator.')

parser.add_argument('--lambd', type=float, default=1e-3, help='trade-off ratio.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')

parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP instead of GNN')

parser.add_argument('--der', type=float, default=0.2, help='Drop edge ratio.')
parser.add_argument('--dfr', type=float, default=0.2, help='Drop feature ratio.')

parser.add_argument("--hid_dim", type=int, default=128, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=128, help='Output layer dim.')

args = parser.parse_args()

dt=args.dataname

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':

    print(args)
    ratio=args.der
    res_dir = '%s-ccassg-mia-white-2-%s' % (args.dataname, ratio)
    graph, feat, labels, num_class, train_idx, val_idx, test_idx,train_edges0,edges_test0 = load_mia_white2(args.dataname,res_dir)
    in_dim = feat.shape[1]

    model = CCA_SSG_white(in_dim, args.hid_dim, args.out_dim, args.n_layers, args.use_mlp)
    model = model.to(args.device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    N = graph.number_of_nodes()

    z1_trains=[]
    z2_trains=[]

    edge_index1_trains = []
    edge_index2_trains = []

    best_valid_loss=-99999999

    sigma=0.4

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        graph1, feat1,aug_list1 = random_aug_white(graph, feat, args.dfr, args.der)
        graph2, feat2,aug_list2 = random_aug_white(graph, feat, args.dfr, args.der)

        graph1 = graph1.add_self_loop()
        graph2 = graph2.add_self_loop()

        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        z1, z2,emb1,emb2 = model(graph1, feat1, graph2, feat2)

        c = th.mm(z1.T, z2)
        c1 = th.mm(z1.T, z1)
        c2 = th.mm(z2.T, z2)

        c = c / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = -th.diagonal(c).sum()
        iden = th.tensor(np.eye(c.shape[0])).to(args.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + args.lambd * (loss_dec1 + loss_dec2)

        device=args.device

        grads = [th.zeros(p.shape).to(device) for p in model.parameters()]
        # print((grads))
        igrad = th.autograd.grad(loss, model.parameters(), retain_graph=True)
        # print(igrad)
        # for i in igrad:
        #     print(i.size())

        # exit()

        l2_norm = th.tensor(0.0).to(device)
        for g in igrad:
            l2_norm += g.norm(2) ** 2
            # l2_norm += g.sum().square().tolist()
        # print('time12:', int(time.time() / 1000))
        l2_norm = l2_norm.sqrt()
        # divisor = max(torch.tensor(1.0).to(args.device), l2_norm)
        for i in range(len(igrad)):
            # grads[i] += igrad[i] / divisor
            grads[i] += igrad[i]

        for i in range(len(grads)):
            # print(grads[i])
            grads[i] += sigma * (th.randn_like(grads[i]).to(device))
            # print(grads[i])
            grads[i].detach_()
            # exit()

        p_list = [p for p in model.parameters()]
        for i in range(len(p_list)):
            p_list[i].grad = grads[i]
            # print(p_list[i].grad)
            p_list[i].grad.detach_()

        loss.backward()
        optimizer.step()
        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))


        patience = 100

        if loss > best_valid_loss:
            best_valid_loss = loss
            trail_count = 0
            best_epoch = epoch
            th.save(model.state_dict(), os.path.join('./checkpoint',
                                                                'tmp',
                                                                f'grace_{dt}_{ratio}_best.pt'))

            z1_trains.append(emb1)
            z2_trains.append(emb2)

            edge_index1_trains.append(aug_list1)
            edge_index2_trains.append(aug_list2)



        else:
            trail_count += 1
            if trail_count > patience:
                print(f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                break

            else:

                z1_trains.append(emb1)
                z2_trains.append(emb2)

                edge_index1_trains.append(aug_list1)
                edge_index2_trains.append(aug_list2)

    model.load_state_dict(th.load(os.path.join('./checkpoint', 'tmp',
                                                              f'grace_{dt}_{ratio}_best.pt')))

    aug1s = edge_index1_trains
    aug2s = edge_index2_trains
    z1_trains_ = np.array(z1_trains)
    z2_trains_ = np.array(z2_trains)

    print("=== Evaluation ===")
    graph = graph.to(args.device)
    graph = graph.remove_self_loop().add_self_loop()
    feat = feat.to(args.device)

    embeds = model.get_embedding(graph, feat)

    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]

    label = labels.to(args.device)

    train_labels = label[train_idx]
    val_labels = label[val_idx]
    test_labels = label[test_idx]

    train_feat = feat[train_idx]
    val_feat = feat[val_idx]
    test_feat = feat[test_idx]

    ''' Linear Evaluation '''
    logreg = LogReg(train_embs.shape[1], num_class)
    opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

    logreg = logreg.to(args.device)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(args.epochs):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = th.argmax(logits, dim=1)
        train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with th.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            out = logreg(embeds)

            val_preds = th.argmax(val_logits, dim=1)
            test_preds = th.argmax(test_logits, dim=1)

            val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                if test_acc > eval_acc:
                    eval_acc = test_acc

            print(
                'Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))

    print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))

    emb_matrix = embeds.cpu().detach().numpy()
    edges_train_all = train_edges0
    edges_test_all = edges_test0
    label = labels.cpu().detach().numpy()
    output_train = out.cpu().detach().numpy()

    # edges_train_inter=np.array(edges_train_inter)
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

    test_edges_sampled = edges_test_all

    z1_trains = z1_trains_
    z2_trains = z2_trains_

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
    edges_mia_pos_idx = np.array(edges_mia_pos)[:, 0] * 99999 + np.array(edges_mia_pos)[:, 1]  # pos testing

    edges_mia_neg = [[min(edge[0], edge[1]), max(edge[0], edge[1])] for edge in edges_mia_neg0]  # neg testing

    edges_mia_neg_idx = np.array(edges_mia_neg)[:, 0] * 99999 + np.array(edges_mia_neg)[:, 1]

    train_edges_sampled_ = [[min(edge[0], edge[1]), max(edge[0], edge[1])] for edge in train_edges_sampled1]
    test_edges_sampled_ = [[min(edge[0], edge[1]), max(edge[0], edge[1])] for edge in edges_test_all]

    train_edges_sampled_idx = np.array(train_edges_sampled_)[:, 0] * 99999 + np.array(train_edges_sampled_)[:, 1]
    test_edges_sampled_idx = np.array(test_edges_sampled_)[:, 0] * 99999 + np.array(test_edges_sampled_)[:, 1]

    train_edges_pos_idx = np.setdiff1d(train_edges_sampled_idx, edges_mia_pos_idx)  # pos training
    train_edges_neg_idx = np.setdiff1d(test_edges_sampled_idx, edges_mia_neg_idx)  # neg training

    print(len(train_edges_sampled_idx), len(test_edges_sampled_idx), len(train_edges_pos_idx), len(train_edges_neg_idx))
    print(len(train_edges_pos_idx), len(train_edges_neg_idx))
    # # exit()
    #
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
    #
    drop1s_pos_idx_test = []
    drop2s_pos_idx_test = []

    for aug_idx in aug1s_idx:
        drop_idx = np.setdiff1d(train_edges_pos_idx, aug_idx)
        drop1s_pos_idx.append(drop_idx)

    for aug_idx in aug2s_idx:
        drop_idx = np.setdiff1d(train_edges_pos_idx, aug_idx)
        drop2s_pos_idx.append(drop_idx)

    # print(drop1s_pos_idx)
    # print(drop2s_pos_idx)

    for aug_idx in aug1s_idx:
        drop_idx = np.setdiff1d(edges_mia_pos_idx, aug_idx)
        drop1s_pos_idx_test.append(drop_idx)

    for aug_idx in aug2s_idx:
        drop_idx = np.setdiff1d(edges_mia_pos_idx, aug_idx)
        drop2s_pos_idx_test.append(drop_idx)

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

    with open('./%s/%s-drop1s_pos_idx_test.txt' % (res_dir, dt), 'w') as f:
        for item in drop1s_pos_idx_test:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    with open('./%s/%s-drop2s_pos_idx_test.txt' % (res_dir, dt), 'w') as f:
        for item in drop2s_pos_idx_test:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    # file_name = './%s/%s-drop1s_pos_idx.txt' % (res_dir, dt)
    # drop1s_pos_idx0 = readedges2(file_name)
    # # print(drop1s_pos_idx)
    #
    # file_name = './%s/%s-drop2s_pos_idx.txt' % (res_dir, dt)
    # drop2s_pos_idx0 = readedges2(file_name)
    #
    # print('####', drop1s_pos_idx0[0])
    #
    # # print(drop2s_pos_idx0[0])
    #
    # # print(drop2s_pos_idx0[0])
    # file_name = './%s/%s-drop1s_pos_idx_test.txt' % (res_dir, dt)
    # drop1s_pos_idx0_test = readedges2(file_name)
    # # print(drop1s_pos_idx)
    #
    # file_name = './%s/%s-drop2s_pos_idx_test.txt' % (res_dir, dt)
    # drop2s_pos_idx0_test = readedges2(file_name)

    drop1s_pos_idx0 = drop1s_pos_idx
    drop2s_pos_idx0 = drop2s_pos_idx
    drop1s_pos_idx0_test = drop1s_pos_idx_test
    drop2s_pos_idx0_test = drop2s_pos_idx_test

    iterations = np.shape(drop2s_pos_idx0)[0]

    # iter_ratios = [0.2, 0.4, 0.6, 0.8, 1]
    iter_ratios = [1]

    results = []
    for iters in iter_ratios:
        iter_ = int(iterations * iters) - 1

        drop1s_pos_idx = drop1s_pos_idx0[0:iter_]
        drop2s_pos_idx = drop2s_pos_idx0[0:iter_]

        drop1s_pos_idx_test = drop1s_pos_idx0_test[0:iter_]
        drop2s_pos_idx_test = drop2s_pos_idx0_test[0:iter_]

        drop1s_pos_idx_ = list(itertools.chain.from_iterable(drop1s_pos_idx))
        drop2s_pos_idx_ = list(itertools.chain.from_iterable(drop2s_pos_idx))

        drop1s_pos_idx_test_ = list(itertools.chain.from_iterable(drop1s_pos_idx_test))
        drop2s_pos_idx_test_ = list(itertools.chain.from_iterable(drop2s_pos_idx_test))

        print(len(drop1s_pos_idx_), len(drop2s_pos_idx_))
        set1 = list(set(drop1s_pos_idx_))
        set2 = list(set(drop2s_pos_idx_))
        print(len(set1), len(set2))
        set0 = list(set(set1 + set2))
        # print(set0)
        print(len(set0))
        print(np.shape(test_edges_sampled)[0])
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

        aug1s_embed = z1_trains
        aug2s_embed = z2_trains

        epoches = np.shape(aug1s_embed)[0]
        idx_epoches = list(range(epoches))

        idx_epoches_all = []
        drop_idx_all = []
        for i in train_edges_pos_idx:

            if i in idx_dic1_.keys():

                drop_idx = idx_dic1_[i]
                idx_epoches_ = list(set(idx_epoches).difference(set(drop_idx)))
                if len(drop_idx) < max(idx_dic1.values()):
                    drop_idx_sample2 = random.sample(idx_epoches_, (epoches - max(idx_dic1.values()) - len(drop_idx)))

                    drop_idx_sample = random.sample(idx_epoches_, (max(idx_dic1.values()) - len(drop_idx)))
                    idx_epoches_ = list(set(idx_epoches_).difference(set(drop_idx_sample)))

                    drop_idx_ = list(drop_idx) + drop_idx_sample2
                else:
                    idx_epoches_ = list(set(idx_epoches_))
                    drop_idx_ = idx_epoches_

            else:
                idx_epoches_ = idx_epoches

                drop_idx_sample = random.sample(idx_epoches_, (max(idx_dic1.values())))

                idx_epoches_ = list(set(idx_epoches).difference(set(drop_idx_sample)))

                drop_idx_ = idx_epoches_

            idx_epoches_all.append(idx_epoches_)

            drop_idx_all.append(drop_idx_)

        set1 = list(set(drop1s_pos_idx_test_))
        set2 = list(set(drop2s_pos_idx_test_))
        print(len(set1), len(set2))
        set0 = list(set(set1 + set2))
        # print(set0)
        print(len(set0))
        print(np.shape(test_edges_sampled)[0])
        # exit()
        idx_dic1_test = dict()
        idx_dic2_test = dict()
        idx_dic1_test_ = dict()
        idx_dic2_test_ = dict()
        for idx in set0:
            idx_dic1_test[idx] = 0
            idx_dic2_test[idx] = 0
            idx_dic1_test_[idx] = []
            idx_dic2_test_[idx] = []

        i = 0
        for idx in drop1s_pos_idx_test:
            for j in idx:
                idx_dic1_test[j] += 1
                idx_dic1_test_[j].append(i)
            i += 1

        i = 0
        for idx in drop2s_pos_idx_test:
            for j in idx:
                idx_dic2_test[j] += 1
                idx_dic2_test_[j].append(i)
            i += 1

        # print(min(idx_dic1.values()),max(idx_dic1.values()))
        # print(min(idx_dic2.values()),max(idx_dic2.values()))

        # print(idx_dic1,idx_dic2)
        idx_dic0_test = []
        for idx in set0:
            idx_dic0_test.append(idx_dic1_test[idx] + idx_dic2_test[idx])
        # print(idx_dic0)
        # print(min(idx_dic0),max(idx_dic0))

        train_edges_pos_test = []
        train_edges_neg_test = []
        for i in edges_mia_pos_idx:
            node1 = int(i / 99999)
            node2 = i % 99999
            train_edges_pos_test.append([node1, node2])

        for i in edges_mia_neg_idx:
            node1 = int(i / 99999)
            node2 = i % 99999
            train_edges_neg_test.append([node1, node2])

        test_edges_pos = np.array(edges_mia_pos)
        test_edges_neg = np.array(edges_mia_neg)

        # epoches=np.shape(aug1s_embed)[0]
        # idx_epoches=list(range(epoches))

        idx_epoches_all_test = []
        # drop_idx_all = []
        for i in edges_mia_pos_idx:

            if i in idx_dic1_test_.keys():

                drop_idx = idx_dic1_test_[i]
                idx_epoches_ = list(set(idx_epoches).difference(set(drop_idx)))
                if len(drop_idx) < max(idx_dic1_test.values()):
                    drop_idx_sample2 = random.sample(idx_epoches_, (epoches - max(idx_dic1.values()) - len(drop_idx)))
                    drop_idx_sample = random.sample(idx_epoches_, (max(idx_dic1_test.values()) - len(drop_idx)))
                    idx_epoches_test_ = list(set(idx_epoches_).difference(set(drop_idx_sample)))
                    drop_idx_ = list(drop_idx) + drop_idx_sample2
                else:
                    idx_epoches_test_ = list(set(idx_epoches_))
                    drop_idx_ = idx_epoches_

            else:
                idx_epoches_ = idx_epoches

                drop_idx_sample = random.sample(idx_epoches_, (max(idx_dic1_test.values())))

                idx_epoches_test_ = list(set(idx_epoches).difference(set(drop_idx_sample)))

                drop_idx_ = idx_epoches_

            idx_epoches_all_test.append(idx_epoches_test_)
            # drop_idx_all.append(drop_idx_)

        idx_epoches_all = np.array(idx_epoches_all)
        drop_idx_all = np.array(drop_idx_all)
        train_edges_pos = np.array(train_edges_pos)
        train_edges_neg = np.array(train_edges_neg)

        idx_epoches_all_test = np.array(idx_epoches_all_test)

        print()

        print('iii', np.shape(train_edges_pos), np.shape(train_edges_neg))

        # idx_epoches_all_neg_train=[]
        # idx_epoches_all_pos_test=[]
        # idx_epoches_all_neg_test=[]
        #
        # for j in range(np.shape(train_edges_neg)[0]):
        #     tmp=random.sample(range(np.shape(aug1s_embed)[0]), (np.shape(idx_epoches_all)[1]))
        #     idx_epoches_all_neg_train.append(tmp)
        #
        #
        # # print('%%%',np.shape(train_edges_neg),np.shape(test_edges_neg),np.shape(test_edges_pos))
        #
        # for j in range(np.shape(test_edges_pos)[0]):
        #     tmp=random.sample(range(np.shape(aug1s_embed)[0]), (np.shape(idx_epoches_all)[1]))
        #     idx_epoches_all_pos_test.append(tmp)
        #
        # for j in range(np.shape(test_edges_neg)[0]):
        #     tmp=random.sample(range(np.shape(aug1s_embed)[0]), (np.shape(idx_epoches_all)[1]))
        #     idx_epoches_all_neg_test.append(tmp)

        # idx_epoches_all_neg_train = np.array(idx_epoches_all_neg_train)
        # idx_epoches_all_pos_test = np.array(idx_epoches_all_pos_test)
        # idx_epoches_all_neg_test =  np.array(idx_epoches_all_neg_test)

        y_train_train = np.concatenate((train_edges_pos, np.ones(np.shape(train_edges_pos)[0]).reshape(-1, 1)), axis=1)
        y_train_test = np.concatenate((train_edges_neg, np.zeros(np.shape(train_edges_neg)[0]).reshape(-1, 1)), axis=1)
        y_test_train = np.concatenate((test_edges_pos, np.ones(np.shape(test_edges_pos)[0]).reshape(-1, 1)), axis=1)
        y_test_test = np.concatenate((test_edges_neg, np.zeros(np.shape(test_edges_neg)[0]).reshape(-1, 1)), axis=1)

        print(np.shape(train_edges_pos), np.shape(idx_epoches_all), np.shape(aug1s_embed))
        pos_train_edge_embs0 = get_edge_embeddings(train_edges_pos, aug1s_embed, idx_epoches_all)
        neg_train_edge_embs0 = get_edge_embeddings(train_edges_neg, aug1s_embed, drop_idx_all)

        # pos_test_edge_embs0 = get_edge_embeddings(test_edges_pos, aug1s_embed,idx_epoches_all_test)
        # neg_test_edge_embs0 = get_edge_embeddings(test_edges_neg, aug1s_embed,idx_epoches_all_test)

        pos_test_edge_embs0 = get_edge_embeddings(test_edges_pos, aug1s_embed, idx_epoches_all)
        neg_test_edge_embs0 = get_edge_embeddings(test_edges_neg, aug1s_embed, drop_idx_all)

        pos_train_edge_embs1 = get_edge_embeddings(train_edges_pos, aug2s_embed, idx_epoches_all)
        neg_train_edge_embs1 = get_edge_embeddings(train_edges_neg, aug2s_embed, drop_idx_all)

        # pos_test_edge_embs1 = get_edge_embeddings(test_edges_pos, aug2s_embed,idx_epoches_all_test)
        # neg_test_edge_embs1 = get_edge_embeddings(test_edges_neg, aug2s_embed,idx_epoches_all_test)

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

        #
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



