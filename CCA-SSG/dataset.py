import numpy as np
import torch as th

import torch
import dgl

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset

import random
import preprocessing
import networkx as nx
import pickle as pkl

def load(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()

    graph = dataset[0]
    citegraph = ['cora', 'citeseer', 'pubmed']
    cograph = ['photo', 'comp', 'cs', 'physics']

    if name in citegraph:
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    if name in cograph:
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num]
        val_idx = idx[train_num:val_num]
        test_idx = idx[val_num:]

        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')

    return graph, feat, labels, num_class, train_idx, val_idx, test_idx

    # edge_mask_rate=0.3
    #
    # edges=graph.edges()
    #
    # g = nx.Graph()
    # g.add_edges_from(edges)
    # adj_sparse = nx.to_scipy_sparse_matrix(g)
    # random.seed(42)
    # train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)
    # adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split
    # # print(adj_train)
    # g_train0 = nx.from_scipy_sparse_matrix(
    #     adj_train)  # new graph object with only non-hidden edges, keep all the original nodes
    #
    # edge_tuples0 = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in g_train0.edges()]
    # # print(edge_tuples0)
    #
    # train_edges0 = set(edge_tuples0)  # initialize train_edges to have all edges
    # train_edges0 = np.array([list(edge_tuple) for edge_tuple in train_edges0])
    # # print(train_edges1)
    #
    #
    # edge_tuples_test0 = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in test_edges]
    #
    # edges_test0 = set(edge_tuples_test0)  # initialize test_edges to have all edges
    # edges_test0 = np.array([list(edge_tuple) for edge_tuple in edges_test0])
    #
    # test_edges0 = edges_test0
    #
    # res_dir = '%s-ccassg-mia-mi' % (name)
    #
    # out = open('%s/%s-edges-train.txt' % (res_dir, name), 'w')
    # for item in train_edges0:
    #     for jtem in item:
    #         out.write(str(jtem) + '\t')
    #     out.write('\n')
    # out.close()
    #
    # out = open('%s/%s-edges-test.txt' % (res_dir, name), 'w')
    # for item in edges_test0:
    #     for jtem in item:
    #         out.write(str(jtem) + '\t')
    #     out.write('\n')
    # out.close()
    #
    # # adj = adj_train
    # #
    # # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    # train_edges_1 = np.concatenate((train_edges0[:, 1].reshape(-1, 1), train_edges0[:, 0].reshape(-1, 1)), axis=1)
    # train_edges_1 = (np.array(train_edges_1))
    # train_edges_2 = (np.array(train_edges0))
    # test_edges_1 = np.concatenate((test_edges0[:, 1].reshape(-1, 1), test_edges0[:, 0].reshape(-1, 1)), axis=1)
    # test_edges_1 = (np.array(test_edges_1))
    # test_edges_2 = (np.array(test_edges0))
    #
    # edges_train_index = np.concatenate((train_edges_1, train_edges_2), axis=0)
    #
    # edges_test_index = np.concatenate((test_edges_1, test_edges_2), axis=0)
    #
    # # graph.edges=edges_train_index
    #
    #
    #
    # if name in citegraph:
    #     train_mask = graph.ndata.pop('train_mask')
    #     val_mask = graph.ndata.pop('val_mask')
    #     test_mask = graph.ndata.pop('test_mask')
    #
    #     train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    #     val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    #     test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    #
    # if name in cograph:
    #     train_ratio = 0.1
    #     val_ratio = 0.1
    #     test_ratio = 0.8
    #
    #     N = graph.number_of_nodes()
    #     train_num = int(N * train_ratio)
    #     val_num = int(N * (train_ratio + val_ratio))
    #
    #     idx = np.arange(N)
    #     np.random.shuffle(idx)
    #
    #     train_idx = idx[:train_num]
    #     val_idx = idx[train_num:val_num]
    #     test_idx = idx[val_num:]
    #
    #     train_idx = th.tensor(train_idx)
    #     val_idx = th.tensor(val_idx)
    #     test_idx = th.tensor(test_idx)
    #
    # num_class = dataset.num_classes
    # feat = graph.ndata.pop('feat')
    # labels = graph.ndata.pop('label')
    #
    # return graph, feat, labels, num_class, train_idx, val_idx, test_idx



def load_mia(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()

    graph = dataset[0]
    citegraph = ['cora', 'citeseer', 'pubmed']
    cograph = ['photo', 'comp', 'cs', 'physics']

    edge_mask_rate=0.3

    edges=graph.edges()
    print(edges)

    e1=edges[0].numpy()
    e2 = edges[1].numpy()
    _edges=np.concatenate((e1.reshape(-1,1),e2.reshape(-1,1)),axis=1)
    edges_=_edges

    g = nx.Graph()
    g.add_edges_from((edges_))
    adj_sparse = nx.to_scipy_sparse_matrix(g)
    random.seed(42)
    train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)
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

    test_edges0 = edges_test0

    res_dir = '%s-ccassg-mia' % (name)

    out = open('%s/%s-edges-train.txt' % (res_dir, name), 'w')
    for item in train_edges0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test.txt' % (res_dir, name), 'w')
    for item in edges_test0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    # adj = adj_train
    #
    # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    train_edges_1 = np.concatenate((train_edges0[:, 1].reshape(-1, 1), train_edges0[:, 0].reshape(-1, 1)), axis=1)
    train_edges_1 = np.transpose(np.array(train_edges_1))
    train_edges_2 = np.transpose(np.array(train_edges0))
    test_edges_1 = np.concatenate((test_edges0[:, 1].reshape(-1, 1), test_edges0[:, 0].reshape(-1, 1)), axis=1)
    test_edges_1 = np.transpose(np.array(test_edges_1))
    test_edges_2 = np.transpose(np.array(test_edges0))

    edges_train_index = np.concatenate((train_edges_1, train_edges_2), axis=1)

    edges_test_index = np.concatenate((test_edges_1, test_edges_2), axis=1)

    print(graph.edges)

    # graph.edges=[]
    # print('@@@@ ',graph.edges)

    # graph.add_edges(edges_train_index)



    if name in citegraph:
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    if name in cograph:
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num]
        val_idx = idx[train_num:val_num]
        test_idx = idx[val_num:]

        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')


    edges_src = torch.from_numpy(edges_train_index[0])
    edges_dst = torch.from_numpy(edges_train_index[1])

    graph0 = dgl.graph((edges_src, edges_dst), num_nodes=g.number_of_nodes())
    graph0.ndata['feat'] = feat
    graph0.ndata['label'] = labels
    # graph0.edata['weight'] = edge_features

    print(feat, labels)
    print(feat.type(), labels.type())
    # exit()

    return graph0, feat, labels, num_class, train_idx, val_idx, test_idx,train_edges0,edges_test0


def load_DE(dt):
    name=dt

    feat_dir = './data/' + dt + '-adj-feat.pkl'

    f2 = open(feat_dir, 'rb')

    adj, ft, labels = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    x = ft
    num_features = np.shape(ft)[1]

    x = torch.from_numpy(np.array(x)).float()

    graph = g

    edge_mask_rate=0.3

    edges=graph.edges()
    print(edges)

    edges_all = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in g.edges()]

    edges_all = set(edges_all)  # initialize train_edges to have all edges
    edges_ = np.array([list(edge_tuple) for edge_tuple in edges_all])


    # e1=edges[0].numpy()
    # e2 = edges[1].numpy()
    # _edges=np.concatenate((e1.reshape(-1,1),e2.reshape(-1,1)),axis=1)
    # edges_=_edges

    g = nx.Graph()
    g.add_edges_from((edges_))
    adj_sparse = nx.to_scipy_sparse_matrix(g)
    random.seed(42)
    train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)
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

    test_edges0 = edges_test0

    res_dir = '%s-ccassg-mia-mi' % (name)

    out = open('%s/%s-edges-train.txt' % (res_dir, name), 'w')
    for item in train_edges0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test.txt' % (res_dir, name), 'w')
    for item in edges_test0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    # adj = adj_train
    #
    # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    train_edges_1 = np.concatenate((train_edges0[:, 1].reshape(-1, 1), train_edges0[:, 0].reshape(-1, 1)), axis=1)
    train_edges_1 = np.transpose(np.array(train_edges_1))
    train_edges_2 = np.transpose(np.array(train_edges0))
    test_edges_1 = np.concatenate((test_edges0[:, 1].reshape(-1, 1), test_edges0[:, 0].reshape(-1, 1)), axis=1)
    test_edges_1 = np.transpose(np.array(test_edges_1))
    test_edges_2 = np.transpose(np.array(test_edges0))

    edges_train_index = np.concatenate((train_edges_1, train_edges_2), axis=1)

    edges_test_index = np.concatenate((test_edges_1, test_edges_2), axis=1)

    print(graph.edges)

    # graph.edges=[]
    # print('@@@@ ',graph.edges)

    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2

    N = graph.number_of_nodes()
    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    train_idx = th.tensor(train_idx)
    val_idx = th.tensor(val_idx)
    test_idx = th.tensor(test_idx)

    num_class = max(labels)+1
    feat = x

    edges_src = torch.from_numpy(edges_train_index[0])
    edges_dst = torch.from_numpy(edges_train_index[1])

    labels = torch.from_numpy(labels).long()


    graph0 = dgl.graph((edges_src, edges_dst), num_nodes=g.number_of_nodes())
    graph0.ndata['feat'] = feat
    graph0.ndata['label'] = labels
    # graph0.edata['weight'] = edge_features

    return graph0, feat, labels, num_class, train_idx, val_idx, test_idx,train_edges0,edges_test0,res_dir


def load_chemistry(dt):
    name=dt

    feat_dir = '../data/' + dt + '-adj-feat.pkl'
    # feat_dir = '/Users/xiulingwang/Downloads/line-master/26-adj-feat.pkl'

    f2 = open(feat_dir, 'rb')

    adj, ft,labels = pkl.load(f2, encoding='latin1')

    # print(ft)
    x=ft

    g = nx.Graph(adj)
    print(g.number_of_nodes(),g.number_of_edges())

    for ii in range(np.shape(x)[0]):
        if np.sum(x[ii])!=0:

            x[ii]=x[ii]/np.sum(x[ii])


    print(set(list(labels)))

    for cls in set(list(labels)):
        print('KKKKK', cls, len(np.where(labels == cls)[0]))

    labels = torch.from_numpy(labels).long()

    x = torch.from_numpy(np.array(x)).float()



    edges_all = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in g.edges()]

    edges_all = set(edges_all)  # initialize train_edges to have all edges
    edges_ = np.array([list(edge_tuple) for edge_tuple in edges_all])


    # e1=edges[0].numpy()
    # e2 = edges[1].numpy()
    # _edges=np.concatenate((e1.reshape(-1,1),e2.reshape(-1,1)),axis=1)
    # edges_=_edges

    g = nx.Graph()

    g.add_edges_from((edges_))

    adj_sparse = nx.to_scipy_sparse_matrix(g)
    random.seed(42)
    train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)
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

    test_edges0 = edges_test0

    res_dir = '%s-ccassg-mia-mi' % (name)

    out = open('%s/%s-edges-train.txt' % (res_dir, name), 'w')
    for item in train_edges0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test.txt' % (res_dir, name), 'w')
    for item in edges_test0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    # adj = adj_train
    #
    # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    train_edges_1 = np.concatenate((train_edges0[:, 1].reshape(-1, 1), train_edges0[:, 0].reshape(-1, 1)), axis=1)
    train_edges_1 = np.transpose(np.array(train_edges_1))
    train_edges_2 = np.transpose(np.array(train_edges0))
    test_edges_1 = np.concatenate((test_edges0[:, 1].reshape(-1, 1), test_edges0[:, 0].reshape(-1, 1)), axis=1)
    test_edges_1 = np.transpose(np.array(test_edges_1))
    test_edges_2 = np.transpose(np.array(test_edges0))

    edges_train_index = np.concatenate((train_edges_1, train_edges_2), axis=1)

    edges_test_index = np.concatenate((test_edges_1, test_edges_2), axis=1)

    print(g.edges)

    # graph.edges=[]
    # print('@@@@ ',graph.edges)

    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2

    N = g.number_of_nodes()
    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    train_idx = th.tensor(train_idx)
    val_idx = th.tensor(val_idx)
    test_idx = th.tensor(test_idx)

    num_class = max(labels)+1
    feat = x

    edges_src = torch.from_numpy(edges_train_index[0])
    edges_dst = torch.from_numpy(edges_train_index[1])

    labels = torch.from_numpy(labels).long()


    graph0 = dgl.graph((edges_src, edges_dst), num_nodes=g.number_of_nodes())
    graph0.ndata['feat'] = feat
    graph0.ndata['label'] = labels
    # graph0.edata['weight'] = edge_features

    return graph0, feat, labels, num_class, train_idx, val_idx, test_idx,train_edges0,edges_test0,res_dir



def load_fb(dt):
    name=dt

    feat_dir = '../data/' + dt + '-adj-feat.pkl'
    # feat_dir = '/Users/xiulingwang/Downloads/line-master/26-adj-feat.pkl'

    f2 = open(feat_dir, 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')

    # print(ft)
    ft=ft


    # gender_idx=0
    # edu_idx=53
    # idx=gender_idx

    # lbs=np.sum(ft[:, idx:idx+1],axis=1)
    # print(ft[:, idx:idx+1])
    # classes = int(np.max(lbs))
    # print(set(list(lbs)))
    # for cls in range(classes+1):
    #     print('KKKKK', cls, len(np.where(lbs == cls)[0]))

    g = nx.Graph(adj)
    print(g.number_of_nodes(),g.number_of_edges())


    if dt=='3980':

        featname_dir = '../data/' + str(dt) + '.featnames'
        # facebook feature map
        f = open(featname_dir)
        featnames = []
        for line in f:
            line = line.strip().split(' ')
            feats = line[1]
            feats = feats.split(';')
            feat = feats[0]
            featnames.append(feat)
        # print(featnames)
        # exit()
        f.close()

        # gender 77, gender 78
        gindex = featnames.index('gender')
        print(gindex)
        x = np.delete(ft, [gindex,gindex+1], axis=1)
        labels = np.sum(ft[:, gindex:gindex+1], axis=1)
        num_features = np.shape(ft)[1] - 2
    elif dt=='combined':
        gender_idx=77
        edu_idx=53
        idx=gender_idx
        gindex=idx
        x = np.delete(ft, [gindex,gindex+1], axis=1)
        labels = np.sum(ft[:, gindex:gindex + 1], axis=1)
        num_features = np.shape(ft)[1] - 2

    elif dt=='dblp-2':
        gindex=0
        x = np.delete(ft, [gindex], axis=1)
        labels = np.sum(ft[:, gindex], axis=1)
        num_features = np.shape(ft)[1] - 1

    elif dt=='pokec':
        gindex=0
    # print(gindex)
        x = np.delete(ft,[gindex],axis=1)
        labels = np.sum(ft[:, gindex], axis=1)
        num_features = np.shape(ft)[1] - 1
    for ii in range(np.shape(x)[0]):
        if np.sum(x[ii])!=0:

            x[ii]=x[ii]/np.sum(x[ii])


    print(set(list(labels)))

    for cls in set(list(labels)):
        print('KKKKK', cls, len(np.where(labels == cls)[0]))

    labels = torch.from_numpy(labels).long()

    x = torch.from_numpy(np.array(x)).float()



    edges_all = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in g.edges()]

    edges_all = set(edges_all)  # initialize train_edges to have all edges
    edges_ = np.array([list(edge_tuple) for edge_tuple in edges_all])


    # e1=edges[0].numpy()
    # e2 = edges[1].numpy()
    # _edges=np.concatenate((e1.reshape(-1,1),e2.reshape(-1,1)),axis=1)
    # edges_=_edges

    g = nx.Graph()

    g.add_edges_from((edges_))

    adj_sparse = nx.to_scipy_sparse_matrix(g)
    random.seed(42)
    train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)
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

    test_edges0 = edges_test0

    res_dir = '%s-ccassg-mia-mi' % (name)

    out = open('%s/%s-edges-train.txt' % (res_dir, name), 'w')
    for item in train_edges0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test.txt' % (res_dir, name), 'w')
    for item in edges_test0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    # adj = adj_train
    #
    # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    train_edges_1 = np.concatenate((train_edges0[:, 1].reshape(-1, 1), train_edges0[:, 0].reshape(-1, 1)), axis=1)
    train_edges_1 = np.transpose(np.array(train_edges_1))
    train_edges_2 = np.transpose(np.array(train_edges0))
    test_edges_1 = np.concatenate((test_edges0[:, 1].reshape(-1, 1), test_edges0[:, 0].reshape(-1, 1)), axis=1)
    test_edges_1 = np.transpose(np.array(test_edges_1))
    test_edges_2 = np.transpose(np.array(test_edges0))

    edges_train_index = np.concatenate((train_edges_1, train_edges_2), axis=1)

    edges_test_index = np.concatenate((test_edges_1, test_edges_2), axis=1)

    print(g.edges)

    # graph.edges=[]
    # print('@@@@ ',graph.edges)

    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2

    N = g.number_of_nodes()
    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    train_idx = th.tensor(train_idx)
    val_idx = th.tensor(val_idx)
    test_idx = th.tensor(test_idx)

    num_class = max(labels)+1
    feat = x

    edges_src = torch.from_numpy(edges_train_index[0])
    edges_dst = torch.from_numpy(edges_train_index[1])

    labels = torch.from_numpy(labels).long()


    graph0 = dgl.graph((edges_src, edges_dst), num_nodes=g.number_of_nodes())
    graph0.ndata['feat'] = feat
    graph0.ndata['label'] = labels
    # graph0.edata['weight'] = edge_features

    return graph0, feat, labels, num_class, train_idx, val_idx, test_idx,train_edges0,edges_test0,res_dir


def load_mia_white(name,res_dir):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()

    graph = dataset[0]

    citegraph = ['cora', 'citeseer', 'pubmed']
    cograph = ['photo', 'comp', 'cs', 'physics']

    edge_mask_rate=0.3

    edges=graph.edges()
    print(edges)

    e1=edges[0].numpy()
    e2 = edges[1].numpy()
    _edges=np.concatenate((e1.reshape(-1,1),e2.reshape(-1,1)),axis=1)
    edges_=_edges

    g = nx.Graph()
    g.add_edges_from((edges_))
    adj_sparse = nx.to_scipy_sparse_matrix(g)
    random.seed(42)
    train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)
    # res_dir = '%s-mvgrl-mia-mi-white-2-%s' % (dt,alpha)
    with open('./%s/%s-train_test_split' % (res_dir, name), 'wb') as f:
        pkl.dump(train_test_split, f)

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

    test_edges0 = edges_test0

    # res_dir = '%s-ccassg-mia' % (name)

    out = open('%s/%s-edges-train.txt' % (res_dir, name), 'w')
    for item in train_edges0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test.txt' % (res_dir, name), 'w')
    for item in edges_test0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    # adj = adj_train
    #
    # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    train_edges_1 = np.concatenate((train_edges0[:, 1].reshape(-1, 1), train_edges0[:, 0].reshape(-1, 1)), axis=1)
    train_edges_1 = np.transpose(np.array(train_edges_1))
    train_edges_2 = np.transpose(np.array(train_edges0))
    test_edges_1 = np.concatenate((test_edges0[:, 1].reshape(-1, 1), test_edges0[:, 0].reshape(-1, 1)), axis=1)
    test_edges_1 = np.transpose(np.array(test_edges_1))
    test_edges_2 = np.transpose(np.array(test_edges0))

    edges_train_index = np.concatenate((train_edges_1, train_edges_2), axis=1)

    edges_test_index = np.concatenate((test_edges_1, test_edges_2), axis=1)

    print(graph.edges)

    # graph.edges=[]
    # print('@@@@ ',graph.edges)

    # graph.add_edges(edges_train_index)



    if name in citegraph:
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    if name in cograph:
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num]
        val_idx = idx[train_num:val_num]
        test_idx = idx[val_num:]

        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')


    edges_src = torch.from_numpy(edges_train_index[0])
    edges_dst = torch.from_numpy(edges_train_index[1])

    graph0 = dgl.graph((edges_src, edges_dst), num_nodes=g.number_of_nodes())
    graph0.ndata['feat'] = feat
    graph0.ndata['label'] = labels
    # graph0.edata['weight'] = edge_features

    print(feat, labels)
    print(feat.type(), labels.type())
    # exit()

    return graph0, feat, labels, num_class, train_idx, val_idx, test_idx,train_edges0,edges_test0

def load_mia_white2(name,res_dir):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()

    graph = dataset[0]
    citegraph = ['cora', 'citeseer', 'pubmed']
    cograph = ['photo', 'comp', 'cs', 'physics']

    edge_mask_rate=0.3

    print()

    edges=graph.edges()
    print(edges)

    e1=edges[0].numpy()
    e2 = edges[1].numpy()
    _edges=np.concatenate((e1.reshape(-1,1),e2.reshape(-1,1)),axis=1)
    edges_=_edges

    feat = graph.ndata.pop('feat')

    g = nx.Graph()
    g.add_edges_from((edges_))
    g.add_nodes_from((list(range(np.shape(feat)[0]))))
    adj_sparse = nx.to_scipy_sparse_matrix(g)
    random.seed(42)
    train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)
    with open('./%s/%s-train_test_split' % (res_dir, name), 'wb') as f:
        pkl.dump(train_test_split, f)


    f2 = open('./%s/%s-train_test_split' % (res_dir, name), 'rb')
    train_test_split = pkl.load(f2, encoding='latin1')

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split
    # print(adj_train)
    g_train0 = nx.from_scipy_sparse_matrix(
        adj_train)  # new graph object with only non-hidden edges, keep all the original nodes

    g_train0.add_nodes_from(list(range(g.number_of_nodes())))

    edge_tuples0 = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in g_train0.edges()]
    # print(edge_tuples0)

    train_edges0 = set(edge_tuples0)  # initialize train_edges to have all edges
    train_edges0 = np.array([list(edge_tuple) for edge_tuple in train_edges0])
    # print(train_edges1)


    edge_tuples_test0 = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in test_edges]

    edges_test0 = set(edge_tuples_test0)  # initialize test_edges to have all edges
    edges_test0 = np.array([list(edge_tuple) for edge_tuple in edges_test0])

    test_edges0 = edges_test0

    # res_dir = '%s-ccassg-mia' % (name)

    out = open('%s/%s-edges-train.txt' % (res_dir, name), 'w')
    for item in train_edges0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test.txt' % (res_dir, name), 'w')
    for item in edges_test0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    # adj = adj_train
    #
    # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    train_edges_1 = np.concatenate((train_edges0[:, 1].reshape(-1, 1), train_edges0[:, 0].reshape(-1, 1)), axis=1)
    train_edges_1 = np.transpose(np.array(train_edges_1))
    train_edges_2 = np.transpose(np.array(train_edges0))
    test_edges_1 = np.concatenate((test_edges0[:, 1].reshape(-1, 1), test_edges0[:, 0].reshape(-1, 1)), axis=1)
    test_edges_1 = np.transpose(np.array(test_edges_1))
    test_edges_2 = np.transpose(np.array(test_edges0))

    edges_train_index = np.concatenate((train_edges_1, train_edges_2), axis=1)

    edges_test_index = np.concatenate((test_edges_1, test_edges_2), axis=1)

    print(graph.edges)

    # graph.edges=[]
    # print('@@@@ ',graph.edges)

    # graph.add_edges(edges_train_index)



    if name in citegraph:
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    if name in cograph:
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num]
        val_idx = idx[train_num:val_num]
        test_idx = idx[val_num:]

        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)

    num_class = dataset.num_classes
    # feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')


    edges_src = torch.from_numpy(edges_train_index[0])
    edges_dst = torch.from_numpy(edges_train_index[1])

    graph0 = dgl.graph((edges_src, edges_dst), num_nodes=g.number_of_nodes())
    graph0.ndata['feat'] = feat
    graph0.ndata['label'] = labels
    # graph0.edata['weight'] = edge_features

    print(feat, labels)
    print(feat.type(), labels.type())
    # exit()

    return graph0, feat, labels, num_class, train_idx, val_idx, test_idx,train_edges0,edges_test0


def load_mia_white_varying_density(name, res_dir,r):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()

    graph = dataset[0]

    citegraph = ['cora', 'citeseer', 'pubmed']
    cograph = ['photo', 'comp', 'cs', 'physics']

    edge_mask_rate = 0.3

    edges = graph.edges()
    print(edges)

    num_edges= graph.number_of_edges()

    print('111',graph.number_of_nodes())


    graph=dgl.remove_edges(graph,np.array(list(range(0,num_edges))))
    dir = '../PyGCL-main/examples/%s-adj-%s' % (name, r)
    f2 = open(dir, 'rb')
    adj = pkl.load(f2, encoding='latin1')
    g0 = nx.Graph(adj)

    graph= dgl.add_nodes(graph, graph.number_of_nodes())

    # graph.add_nodes_from(list(range(graph.number_of_nodes())))

    graph = dgl.add_edges(graph,edges[0].numpy(),edges[1].numpy())
    graph = dgl.add_edges(graph, edges[1].numpy(), edges[0].numpy())

    # graph.add_edges_from(list(eds))

    edges = graph.edges()

    e1 = edges[0].numpy()
    e2 = edges[1].numpy()
    _edges = np.concatenate((e1.reshape(-1, 1), e2.reshape(-1, 1)), axis=1)
    edges_ = _edges

    g = nx.Graph()
    g.add_edges_from((edges_))
    adj_sparse = nx.to_scipy_sparse_matrix(g)
    random.seed(42)

    train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)

    with open('./%s/%s-train_test_split' % (res_dir, name), 'wb') as f:
        pkl.dump(train_test_split, f)

    # f2 = open('./%s/%s-train_test_split' % (res_dir, name), 'rb')
    # train_test_split = pkl.load(f2, encoding='latin1')

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

    test_edges0 = edges_test0

    # res_dir = '%s-ccassg-mia' % (name)

    out = open('%s/%s-edges-train.txt' % (res_dir, name), 'w')
    for item in train_edges0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test.txt' % (res_dir, name), 'w')
    for item in edges_test0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    # adj = adj_train
    #
    # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    train_edges_1 = np.concatenate((train_edges0[:, 1].reshape(-1, 1), train_edges0[:, 0].reshape(-1, 1)), axis=1)
    train_edges_1 = np.transpose(np.array(train_edges_1))
    train_edges_2 = np.transpose(np.array(train_edges0))
    test_edges_1 = np.concatenate((test_edges0[:, 1].reshape(-1, 1), test_edges0[:, 0].reshape(-1, 1)), axis=1)
    test_edges_1 = np.transpose(np.array(test_edges_1))
    test_edges_2 = np.transpose(np.array(test_edges0))

    edges_train_index = np.concatenate((train_edges_1, train_edges_2), axis=1)

    edges_test_index = np.concatenate((test_edges_1, test_edges_2), axis=1)

    print(graph.edges)

    # graph.edges=[]
    # print('@@@@ ',graph.edges)

    # graph.add_edges(edges_train_index)



    if name in citegraph:
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    if name in cograph:
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num]
        val_idx = idx[train_num:val_num]
        test_idx = idx[val_num:]

        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    print(type(feat))
    # labels = graph.ndata.pop('label')
    # print(labels)

    dir = '../PyGCL-main/examples/%s-adj-ft' % (name)
    f2 = open(dir, 'rb')
    _,feat,labels = pkl.load(f2, encoding='latin1')
    feat=torch.from_numpy(feat)
    labels = torch.from_numpy(labels)

    edges_src = torch.from_numpy(edges_train_index[0])
    edges_dst = torch.from_numpy(edges_train_index[1])

    print(g.number_of_nodes(),np.shape(feat))

    graph0 = dgl.graph((edges_src, edges_dst), num_nodes=g.number_of_nodes())
    graph0.ndata['feat'] = feat
    graph0.ndata['label'] = labels
    # graph0.edata['weight'] = edge_features

    print(feat, labels)
    print(feat.type(), labels.type())
    # exit()

    return graph0, feat, labels, num_class, train_idx, val_idx, test_idx, train_edges0, edges_test0


