import sys

import numpy as np

from sklearn.metrics import f1_score,accuracy_score


import pandas as pd

import random
import os
import pickle as pk

import itertools

def readedges(file_name):
    file = open(file_name)

    dataMat = []
    for line in file.readlines():
        curLine = line.strip().split('\t')
        floatLine = list(map(int, curLine))
        # print(floatLine)
        dataMat.append(floatLine)

    embeddings = np.array(dataMat,dtype='int')

    return embeddings


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
def add_laplace_noise(data_list, u=0, b=2):
    laplace_noise = np.random.laplace(u, b, np.shape(data_list))
    return laplace_noise + data_list

def get_edge_embeddings(edge_list, emb_matrixs,idx_epoches_all ):
    u = 0
    b = 1
    emb_matrixs = add_laplace_noise(np.array(emb_matrixs), u, b)

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

results=[]
dt='3980'

# rats=[0.2,0.4,0.6,0.8]
rats=[0.2]
for rat in rats:

    res_dir = '%s-grace-mia-white-2-%s' % (dt, rat)


    file_name='%s/%s-edges-train.txt' % (res_dir, dt)
    train_edges=readedges(file_name)

    file_name='%s/%s-edges-test.txt' % (res_dir, dt)
    test_edges=readedges(file_name)

    file_name='%s/%s-edges-train_sampled.txt' % (res_dir, dt)
    train_edges_sampled = readedges(file_name)

    file_name ='%s/%s-edges-test_sampled.txt' % (res_dir, dt)
    test_edges_sampled = readedges(file_name)

    f2 = open('./%s/%s-aug1.pkl' % (res_dir, dt), 'rb')
    aug1s = pk.load(f2, encoding='latin1')

    f2 = open('./%s/%s-aug2.pkl' % (res_dir, dt), 'rb')
    aug2s = pk.load(f2, encoding='latin1')


    f2 = open('./%s/%s-aug1-embed.pkl' % (res_dir, dt), 'rb')
    aug1s_embed = pk.load(f2, encoding='latin1')

    f2 = open('./%s/%s-aug2-embed.pkl' % (res_dir, dt), 'rb')
    aug2s_embed = pk.load(f2, encoding='latin1')

    #name = ['y_score', 'y_test_grd', 'node1', 'node2']
    graph_path="{}/{}-embed-mlp_sim0.csv".format(res_dir,dt)
    data = pd.read_csv(graph_path)
    edges = data.values.tolist()
    edges=np.array(edges,dtype='int')
    edges_mia = [(min(edge[3], edge[4]), max(edge[3], edge[4]),edge[2]) for edge in edges]
    edges_mia = set(edges_mia)  # initialize test_edges to have all edges
    edges_mia = np.array([list(edge_tuple) for edge_tuple in edges_mia])
    print('###',np.shape(edges_mia))
    # print(edges_mia)


    edges_mia0=np.array(edges_mia)[:,0:2]

    edges_mia=np.array(edges_mia)
    index_pos=np.where(edges_mia[:,2]==1)[0]
    index_neg=np.where(edges_mia[:,2]==0)[0]

    print(len(index_pos),len(index_neg))

    edges_mia_pos0=edges_mia[index_pos]
    edges_mia_neg0=edges_mia[index_neg]

    edges_mia_pos = [[min(edge[0], edge[1]), max(edge[0], edge[1])]for edge in edges_mia_pos0]
    print(np.shape(edges_mia_pos))
    edges_mia_pos_idx=np.array(edges_mia_pos)[:,0]*99999+np.array(edges_mia_pos)[:,1]

    edges_mia_neg= [[min(edge[0], edge[1]), max(edge[0], edge[1])]for edge in edges_mia_neg0]

    edges_mia_neg_idx=np.array(edges_mia_neg)[:,0]*99999+np.array(edges_mia_neg)[:,1]


    train_edges_sampled_=[[min(edge[0], edge[1]), max(edge[0], edge[1])]for edge in train_edges_sampled]
    test_edges_sampled_=[[min(edge[0], edge[1]), max(edge[0], edge[1])]for edge in test_edges_sampled]

    train_edges_sampled_idx=np.array(train_edges_sampled_)[:,0]*99999+np.array(train_edges_sampled_)[:,1]
    test_edges_sampled_idx=np.array(test_edges_sampled_)[:,0]*99999+np.array(test_edges_sampled_)[:,1]


    train_edges_pos_idx=np.setdiff1d(train_edges_sampled_idx, edges_mia_pos_idx)
    train_edges_neg_idx=np.setdiff1d(test_edges_sampled_idx, edges_mia_neg_idx)

    print(len(train_edges_sampled_idx),len(test_edges_sampled_idx),len(train_edges_pos_idx),len(train_edges_neg_idx))
    print(len(train_edges_pos_idx),len(train_edges_neg_idx))
    # # exit()
    #
    aug1s_idx=[]
    for aug in aug1s:
        # print(aug,np.shape(aug))
        aug=aug.T
        aug_=[[min(edge[0], edge[1]), max(edge[0], edge[1])] for edge in aug]
        aug_idx=np.array(aug_)[:,0]*99999+np.array(aug_)[:,1]
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
    drop1s_pos_idx=[]
    drop2s_pos_idx=[]

    for aug_idx in aug1s_idx:
        drop_idx=np.setdiff1d(train_edges_pos_idx,aug_idx)
        drop1s_pos_idx.append(drop_idx)

    for aug_idx in aug2s_idx:
        drop_idx=np.setdiff1d(train_edges_pos_idx,aug_idx)
        drop2s_pos_idx.append(drop_idx)

    # print(drop1s_pos_idx)
    # print(drop2s_pos_idx)


    with open('./%s/%s-drop1s_pos_idx.txt' % (res_dir,dt), 'w') as f:
        for item in drop1s_pos_idx:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    with open('./%s/%s-drop2s_pos_idx.txt' % (res_dir,dt), 'w') as f:
        for item in drop2s_pos_idx:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    file_name='./%s/%s-drop1s_pos_idx.txt' % (res_dir,dt)
    drop1s_pos_idx0=readedges2(file_name)
    # print(drop1s_pos_idx)

    file_name='./%s/%s-drop2s_pos_idx.txt' % (res_dir,dt)
    drop2s_pos_idx0=readedges2(file_name)

    # print('####',drop1s_pos_idx0[0])

    # print(drop2s_pos_idx0[0])

    # print(drop2s_pos_idx0[0])


    iterations=np.shape(drop2s_pos_idx0)[0]

    # iter_ratios=[0.2,0.4,0.6,0.8,1]
    iter_ratios = [1]


    # results=[]
    for iters in iter_ratios:
        iter_=int(iterations*iters)-1

        drop1s_pos_idx=drop1s_pos_idx0[0:iter_]
        drop2s_pos_idx=drop2s_pos_idx0[0:iter_]

        drop1s_pos_idx_=list(itertools.chain.from_iterable(drop1s_pos_idx))
        drop2s_pos_idx_=list(itertools.chain.from_iterable(drop2s_pos_idx))

        print(len(drop1s_pos_idx_),len(drop2s_pos_idx_))
        set1=list(set(drop1s_pos_idx_))
        set2=list(set(drop2s_pos_idx_))
        print(len(set1),len(set2))
        set0=list(set(set1+set2))
        # print(set0)
        print(len(set0))
        print(np.shape(test_edges_sampled)[0])
        # exit()
        idx_dic1=dict()
        idx_dic2=dict()
        idx_dic1_=dict()
        idx_dic2_=dict()
        for idx in set0:
            idx_dic1[idx]=0
            idx_dic2[idx] = 0
            idx_dic1_[idx]=[]
            idx_dic2_[idx] = []

        i=0
        for idx in drop1s_pos_idx:
            for j in idx:
                idx_dic1[j]+=1
                idx_dic1_[j].append(i)
            i+=1

        i=0
        for idx in drop2s_pos_idx:
            for j in idx:
                idx_dic2[j]+=1
                idx_dic2_[j].append(i)
            i += 1

        print(min(idx_dic1.values()),max(idx_dic1.values()))
        print(min(idx_dic2.values()),max(idx_dic2.values()))

        # print(idx_dic1,idx_dic2)
        idx_dic0=[]
        for idx in set0:
            idx_dic0.append(idx_dic1[idx]+idx_dic2[idx])
        # print(idx_dic0)
        print(min(idx_dic0),max(idx_dic0))

        train_edges_pos=[]
        train_edges_neg=[]
        for i in train_edges_pos_idx:
            node1=int(i/99999)
            node2=i%99999
            train_edges_pos.append([node1,node2])

        for i in train_edges_neg_idx:
            node1=int(i/99999)
            node2=i%99999
            train_edges_neg.append([node1,node2])

        test_edges_pos=np.array(edges_mia_pos)
        test_edges_neg=np.array(edges_mia_neg)

        epoches=np.shape(aug1s_embed)[0]
        idx_epoches=list(range(epoches))

        idx_epoches_all = []
        drop_idx_all=[]

        for i in train_edges_pos_idx:

            if i in idx_dic1_.keys():###drop index

                drop_idx = idx_dic1_[i]
                # drop_idx_all.append(drop_idx)
                idx_epoches_ = list(set(idx_epoches).difference(set(drop_idx)))
                if len(drop_idx) < max(idx_dic1.values()):
                    # print(epoches,max(idx_dic1.values()),len(drop_idx))
                    # print(epoches-max(idx_dic1.values()) - len(drop_idx))
                    drop_idx_sample2 = random.sample(idx_epoches_, (epoches-max(idx_dic1.values()) - len(drop_idx)))
                    drop_idx_sample = random.sample(idx_epoches_, (max(idx_dic1.values()) - len(drop_idx)))
                    idx_epoches_ = list(set(idx_epoches_).difference(set(drop_idx_sample)))

                    drop_idx_=list(drop_idx)+drop_idx_sample2

                    # print('111', len(drop_idx_))

                else:
                    idx_epoches_ = list(set(idx_epoches_))
                    drop_idx_=idx_epoches_
                    # print('222', len(drop_idx_))



            else:
                idx_epoches_ = idx_epoches
                drop_idx_sample = random.sample(idx_epoches_, (max(idx_dic1.values())))

                idx_epoches_ = list(set(idx_epoches).difference(set(drop_idx_sample)))
                drop_idx_ = idx_epoches_

                # print('333',len(drop_idx_))

            idx_epoches_all.append(idx_epoches_)
            drop_idx_all.append(drop_idx_)

        idx_epoches_all=np.array(idx_epoches_all)
        drop_idx_all=np.array(drop_idx_all)
        train_edges_pos=np.array(train_edges_pos)
        train_edges_neg=np.array(train_edges_neg)

        y_train_train=np.concatenate((train_edges_pos,np.ones(np.shape(train_edges_pos)[0]).reshape(-1,1)),axis=1)
        y_train_test=np.concatenate((train_edges_neg,np.zeros(np.shape(train_edges_neg)[0]).reshape(-1,1)),axis=1)
        y_test_train=np.concatenate((test_edges_pos,np.ones(np.shape(test_edges_pos)[0]).reshape(-1,1)),axis=1)
        y_test_test=np.concatenate((test_edges_neg,np.zeros(np.shape(test_edges_neg)[0]).reshape(-1,1)),axis=1)

        print(np.shape(train_edges_pos),np.shape(idx_epoches_all),np.shape(drop_idx_all),np.shape(aug1s_embed))
        pos_train_edge_embs0 = get_edge_embeddings(train_edges_pos, aug1s_embed,idx_epoches_all)
        neg_train_edge_embs0 = get_edge_embeddings(train_edges_neg, aug1s_embed,drop_idx_all)

        pos_test_edge_embs0 = get_edge_embeddings(test_edges_pos, aug1s_embed,idx_epoches_all)
        neg_test_edge_embs0 = get_edge_embeddings(test_edges_neg, aug1s_embed,drop_idx_all)

        pos_train_edge_embs1 = get_edge_embeddings(train_edges_pos, aug2s_embed, idx_epoches_all)
        neg_train_edge_embs1= get_edge_embeddings(train_edges_neg, aug2s_embed, drop_idx_all)

        pos_test_edge_embs1 = get_edge_embeddings(test_edges_pos, aug2s_embed, idx_epoches_all)
        neg_test_edge_embs1 = get_edge_embeddings(test_edges_neg, aug2s_embed, drop_idx_all)


        X_train = np.concatenate((pos_train_edge_embs0 ,neg_train_edge_embs0), axis=0)
        X_test = np.concatenate((pos_test_edge_embs0 , neg_test_edge_embs0), axis=0)
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



        X_train = np.concatenate((pos_train_edge_embs1 ,neg_train_edge_embs1), axis=0)
        X_test = np.concatenate((pos_test_edge_embs1 , neg_test_edge_embs1), axis=0)
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

        pos_train_edge_embs1 = np.concatenate((pos_train_edge_embs0 ,pos_train_edge_embs1), axis=1)
        neg_train_edge_embs1 = np.concatenate((neg_train_edge_embs0 ,neg_train_edge_embs1), axis=1)

        pos_test_edge_embs1 = np.concatenate((pos_test_edge_embs0 ,pos_test_edge_embs1), axis=1)
        neg_test_edge_embs1 = np.concatenate((neg_test_edge_embs0 ,neg_test_edge_embs1), axis=1)

        X_train = np.concatenate((pos_train_edge_embs1 ,neg_train_edge_embs1), axis=0)
        X_test = np.concatenate((pos_test_edge_embs1 , neg_test_edge_embs1), axis=0)
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
result_all.to_csv("{}/results_all-lap.csv".format(res_dir))





