import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import torch
import random
import preprocessing
import pickle as pk


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data_mia(dataset_str,drop_edge_rate_1 ,drop_feature_rate_1):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))


    # g = nx.Graph()
    # adj_sparse=nx.from_scipy_sparse_matrix(adj)
    # adj_sparse = nx.to_scipy_sparse_matrix(g)
    # print('***',adj,y)
    # exit()
    if dataset_str=='citeseer':
        dt0='Citeseer'
    else:
        dt0='Cora'

    adj_sparse=adj
    random.seed(42)
    # train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)
    f2 = open('/Wang-ds/xwang193/PyGCL-main/examples/%s-0.2-grace-mia-mi-white-2-nofeature-perturb/%s-train_test_split' % (dt0,dt0), 'rb')
    train_test_split = pk.load(f2, encoding='latin1')


    res_dir = '%s-merit-mia-white-2-nodiffusion-%s-%s' % (dataset_str,drop_edge_rate_1 ,drop_feature_rate_1)
    with open('./%s/%s-train_test_split' % (res_dir, dt0), 'wb') as f:
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

    out = open('%s/%s-edges-train.txt' % (res_dir, dataset_str), 'w')
    for item in train_edges0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test.txt' % (res_dir, dataset_str), 'w')
    for item in edges_test0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    # # adj = adj_train
    # #
    # # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    # train_edges_1 = np.concatenate((train_edges0[:, 1].reshape(-1, 1), train_edges0[:, 0].reshape(-1, 1)), axis=1)
    # train_edges_1 = np.transpose(np.array(train_edges_1))
    # train_edges_2 = np.transpose(np.array(train_edges0))
    # # loop_nodes=np.arange(0,g.number_of_nodes())
    # # train_edges_3=np.concatenate((loop_nodes.reshape(-1,1),loop_nodes.reshape(-1,1)),axis=1)
    # # train_edges_3 = np.transpose(np.array(train_edges_3))
    #
    # edges_train_index = np.concatenate((train_edges_1, train_edges_2), axis=1)
    #
    # adj_idx=edges_train_index




    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    # idx_all=range(y)

    # print(adj_train)

    return adj_train, features, labels, idx_train, idx_val, idx_test,train_edges0,edges_test0,res_dir


def load_data_mia2(dataset_str,res_dir):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()

    # print(features)
    # print(features.type())
    # exit()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))


    # g = nx.Graph()
    # adj_sparse=nx.from_scipy_sparse_matrix(adj)
    # adj_sparse = nx.to_scipy_sparse_matrix(g)
    # print('***',adj,y)
    # exit()
    if dataset_str=='citeseer':
        dt0='Citeseer'
    else:
        dt0='Cora'

    adj_sparse=adj
    random.seed(42)
    # train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)
    # f2 = open('/Wang-ds/xwang193/PyGCL-main/examples/%s-0.2-grace-mia-mi-white-2-nofeature-perturb/%s-train_test_split' % (dt0,dt0), 'rb')
    # train_test_split = pk.load(f2, encoding='latin1')


    # res_dir = '%s-merit-mia-white-2-nodiffusion-%s-%s' % (dataset_str,drop_edge_rate_1 ,drop_feature_rate_1)
    # with open('./%s/%s-train_test_split' % (res_dir, dt0), 'wb') as f:
    #     pk.dump(train_test_split, f)

    f2 = open(
        './%s/%s-train_test_split' % (res_dir, dt0), 'rb')
    train_test_split = pk.load(f2, encoding='latin1')

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

    out = open('%s/%s-edges-train.txt' % (res_dir, dataset_str), 'w')
    for item in train_edges0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test.txt' % (res_dir, dataset_str), 'w')
    for item in edges_test0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    # # adj = adj_train
    # #
    # # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    # train_edges_1 = np.concatenate((train_edges0[:, 1].reshape(-1, 1), train_edges0[:, 0].reshape(-1, 1)), axis=1)
    # train_edges_1 = np.transpose(np.array(train_edges_1))
    # train_edges_2 = np.transpose(np.array(train_edges0))
    # # loop_nodes=np.arange(0,g.number_of_nodes())
    # # train_edges_3=np.concatenate((loop_nodes.reshape(-1,1),loop_nodes.reshape(-1,1)),axis=1)
    # # train_edges_3 = np.transpose(np.array(train_edges_3))
    #
    # edges_train_index = np.concatenate((train_edges_1, train_edges_2), axis=1)
    #
    # adj_idx=edges_train_index




    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    # idx_all=range(y)

    # print(labels)
    # exit()

    return adj_train, features, labels, idx_train, idx_val, idx_test,train_edges0,edges_test0


def load_data_mia2_fb(dataset_str,res_dir):
    # num_nodes = 1600
    # num_feats = 1283

    feat_dir = '3980-adj-feat.pkl'
    # feat_dir = '/Users/xiulingwang/Downloads/line-master/26-adj-feat.pkl'

    f2 = open(feat_dir, 'rb')

    adj, ft = pk.load(f2, encoding='latin1')

    print(np.shape(ft))

    # ft=ft-1


    gender_idx=77
    edu_idx=53

    lbs=np.sum(ft[:, edu_idx:edu_idx+1],axis=1)
    print(ft[:, edu_idx:edu_idx+1])
    classes = int(np.max(lbs))
    print(set(list(lbs)))
    for cls in range(classes+1):
        print('KKKKK', cls, len(np.where(lbs == cls)[0]))

    g = nx.Graph(adj)


    featname_dir = '3980.featnames'
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

    x = np.delete(ft,[gindex],axis=1)
    for ii in range(np.shape(x)[0]):
        if np.sum(x[ii])!=0:

            x[ii]=x[ii]/np.sum(x[ii])
    num_features = np.shape(ft)[1]
    labels=np.sum(ft[:, gindex:gindex+1],axis=1)
    print(labels)

    lbs=[]

    for lb in labels:
        if lb==0:
            lbs.append([1,0])
        else:
            lbs.append([0,1])
    lbs=np.array(lbs)


    g = nx.Graph(adj)


    # adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    random.seed(42)
    train_test_split = preprocessing.mask_test_edges(adj, test_frac=.3, val_frac=0)

    # res_dir = '%s-merit-mia-white-2-nodiffusion-%s-%s' % (dataset_str,drop_edge_rate_1 ,drop_feature_rate_1)
    with open('./%s/3980-train_test_split' % (res_dir), 'wb') as f:
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

    out = open('%s/%s-edges-train.txt' % (res_dir, dataset_str), 'w')
    for item in train_edges0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test.txt' % (res_dir, dataset_str), 'w')
    for item in edges_test0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    # build symmetric adjacency matrix

    idx= np.arange(num_features)
    idx=np.delete(idx,gindex)

    features = ft[:,idx]
    # print(np.shape(features))
    # exit()
    features = sp.coo_matrix(features, dtype=np.float32).tolil()
    # print(features)


    idx_test = range(0,int(len(labels)*0.2))
    idx_val = range(int(len(labels)*0.2), int(len(labels)*0.3))
    idx_train = range(int(len(labels)*0.2),len(labels))


    return adj_train, features, lbs, idx_train, idx_val, idx_test,train_edges0,edges_test0

def sparse_to_tuple(sparse_mx, insert_batch=False):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)