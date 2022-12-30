import numpy as np
import scipy.sparse as sp
import networkx as nx

import torch

from torch import nn
from torch.optim import Adam
import torch.nn.functional as F


# Convert sparse matrix to tuple
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


# Get normalized adjacency matrix: A_norm
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


# Prepare feed-dict for Tensorflow session
def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


# Perform train-test split
# Takes in adjacency matrix in sparse format
# Returns: adj_train, train_edges, val_edges, val_edges_false,
# test_edges, test_edges_false
def mask_test_edges(adj, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=False):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    if verbose == True:
        print('preprocessing...')

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_matrix(adj)
    orig_num_cc = nx.number_connected_components(g)

    adj_triu = sp.triu(adj)  # upper triangular portion of adj matrix
    adj_tuple = sparse_to_tuple(adj_triu)  # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0]  # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    num_test = int(np.floor(edges.shape[0] * test_frac))  # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac))  # controls how alrge the validation set should be

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples)  # initialize train_edges to have all edges
    test_edges = set()
    val_edges = set()

    if verbose == True:
        print('generating test/val sets...')

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print('creating false test edges...')

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if verbose == True:
        print('creating false val edges...')

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
                        false_edge in test_edges_false or \
                        false_edge in val_edges_false:
            continue

        val_edges_false.add(false_edge)

    if verbose == True:
        print('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false,
        # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
                        false_edge in test_edges_false or \
                        false_edge in val_edges_false or \
                        false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose == True:
        print('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print('Done with train-test split!')
        print('')

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, \
           val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_edges2(adj, test_frac=.1, val_frac=.05, prevent_disconnect=False, verbose=False):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    if verbose == True:
        print('preprocessing...')

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_matrix(adj)
    orig_num_cc = nx.number_connected_components(g)

    adj_triu = sp.triu(adj)  # upper triangular portion of adj matrix
    adj_tuple = sparse_to_tuple(adj_triu)  # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0]  # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    num_test = int(np.floor(edges.shape[0] * test_frac))  # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac))  # controls how alrge the validation set should be

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples)  # initialize train_edges to have all edges
    test_edges = set()
    val_edges = set()

    if verbose == True:
        print('generating test/val sets...')

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        g.remove_edge(node1, node2)
        # if prevent_disconnect == True:
        #     if nx.number_connected_components(g) > orig_num_cc:
        #         g.add_edge(node1, node2)
        #         continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print('creating false test edges...')

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if verbose == True:
        print('creating false val edges...')

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
                        false_edge in test_edges_false or \
                        false_edge in val_edges_false:
            continue

        val_edges_false.add(false_edge)

    if verbose == True:
        print('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false,
        # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
                        false_edge in test_edges_false or \
                        false_edge in val_edges_false or \
                        false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose == True:
        print('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print('Done with train-test split!')
        print('')

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, \
           val_edges, val_edges_false, test_edges, test_edges_false




# Perform train-test split
# Takes in adjacency matrix in sparse format (from a directed graph)
# Returns: adj_train, train_edges, val_edges, val_edges_false,
# test_edges, test_edges_false
def mask_test_edges_directed(adj, test_frac=.1, val_frac=.05,
                             prevent_disconnect=True, verbose=False, false_edge_sampling='iterative'):
    if verbose == True:
        print('preprocessing...')

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    # Convert to networkx graph to calc num. weakly connected components
    g = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
    orig_num_wcc = nx.number_weakly_connected_components(g)

    adj_tuple = sparse_to_tuple(adj)  # (coords, values, shape)
    edges = adj_tuple[0]  # List of ALL edges (either direction)
    edge_pairs = [(edge[0], edge[1]) for edge in edges]  # store edges as list of tuples (from_node, to_node)

    num_test = int(np.floor(edges.shape[0] * test_frac))  # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac))  # controls how alrge the validation set should be
    num_train = len(edge_pairs) - num_test - num_val  # num train edges

    all_edge_set = set(edge_pairs)
    train_edges = set(edge_pairs)  # init train_edges to have all edges
    test_edges = set()  # init test_edges as empty set
    val_edges = set()  # init val edges as empty set

    ### ---------- TRUE EDGES ---------- ###
    # Shuffle and iterate over all edges
    np.random.shuffle(edge_pairs)

    # get initial bridge edges
    bridge_edges = set(nx.bridges(nx.to_undirected(g)))

    if verbose:
        print('creating true edges...')

    for ind, edge in enumerate(edge_pairs):
        node1, node2 = edge[0], edge[1]

        # Recalculate bridges every ____ iterations to relatively recent
        if ind % 10000 == 0:
            bridge_edges = set(nx.bridges(nx.to_undirected(g)))

            # Don't sample bridge edges to increase likelihood of staying connected
        if (node1, node2) in bridge_edges or (node2, node1) in bridge_edges:
            continue

        # If removing edge would disconnect the graph, backtrack and move on
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if not nx.is_weakly_connected(g):
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)
            if len(test_edges) % 10000 == 0 and verbose == True:
                print('Current num test edges: ', len(test_edges))

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)
            if len(val_edges) % 10000 == 0 and verbose == True:
                print('Current num val edges: ', len(val_edges))

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    # Check that enough test/val edges were found
    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    # Print stats for largest remaining WCC
    print('Num WCC: ', nx.number_weakly_connected_components(g))
    largest_wcc_set = max(nx.weakly_connected_components(g), key=len)
    largest_wcc = g.subgraph(largest_wcc_set)
    print('Largest WCC num nodes: ', largest_wcc.number_of_nodes())
    print('Largest WCC num edges: ', largest_wcc.number_of_edges())

    if prevent_disconnect == True:
        assert nx.number_weakly_connected_components(g) == orig_num_cc

    # Fraction of edges with both endpoints in largest WCC
    def frac_edges_in_wcc(edge_set):
        num_wcc_contained_edges = 0.0
        num_total_edges = 0.0
        for edge in edge_set:
            num_total_edges += 1
            if edge[0] in largest_wcc_set and edge[1] in largest_wcc_set:
                num_wcc_contained_edges += 1
        frac_in_wcc = num_wcc_contained_edges / num_total_edges
        return frac_in_wcc

    # Check what percentage of edges have both endpoints in largest WCC
    print('Fraction of train edges with both endpoints in L-WCC: ', frac_edges_in_wcc(train_edges))
    print('Fraction of test edges with both endpoints in L-WCC: ', frac_edges_in_wcc(test_edges))
    print('Fraction of val edges with both endpoints in L-WCC: ', frac_edges_in_wcc(val_edges))

    # Ignore edges with endpoint not in largest WCC
    print('Removing edges with either endpoint not in L-WCC from train-test split...')
    train_edges = {edge for edge in train_edges if edge[0] in largest_wcc_set and edge[1] in largest_wcc_set}
    test_edges = {edge for edge in test_edges if edge[0] in largest_wcc_set and edge[1] in largest_wcc_set}
    val_edges = {edge for edge in val_edges if edge[0] in largest_wcc_set and edge[1] in largest_wcc_set}

    ### ---------- FALSE EDGES ---------- ###

    # Initialize empty sets
    train_edges_false = set()
    test_edges_false = set()
    val_edges_false = set()


    # Generate candidate false edges (from g-complement) and iterate through them
    if false_edge_sampling == 'iterative':
        if verbose == True:
            print('preparing complement adjacency matrix...')

        # Sample false edges from G-complement, instead of randomly generating edges
        # g_complement = nx.complement(g)
        adj_complement = 1 - adj.toarray()  # flip 0's, 1's in adjacency matrix
        np.fill_diagonal(adj_complement, val=0)  # set diagonals to 0

        # 2 numpy arrays indicating x, y coords in adj_complement
        # WARNING: This line can use up a lot of RAM depending on 'adj' size
        idx1, idx2 = np.where(adj_complement == 1)

        edges_false = np.stack((idx1, idx2), axis=-1)  # stack arrays into coord pairs.
        edge_pairs_false = [(edge[0], edge[1]) for false_edge in edges_false]

        # Shuffle and iterate over false edges
        np.random.shuffle(edge_pairs_false)
        if verbose == True:
            print('adding candidate false edges to false edge sets...')
        for false_edge in edge_pairs_false:
            # Fill train_edges_false first
            if len(train_edges_false) < len(train_edges):
                train_edges_false.add(false_edge)
                if len(train_edges_false) % 100000 == 0 and verbose == True:
                    print('Current num false train edges: ', len(train_edges_false))

            # Fill test_edges_false next
            elif len(test_edges_false) < len(test_edges):
                test_edges_false.add(false_edge)
                if len(test_edges_false) % 100000 == 0 and verbose == True:
                    print('Current num false test edges: ', len(test_edges_false))

            # Fill val_edges_false last
            elif len(val_edges_false) < len(val_edges):
                val_edges_false.add(false_edge)
                if len(val_edges_false) % 100000 == 0 and verbose == True:
                    print('Current num false val edges: ', len(val_edges_false))

            # All sets filled --> break
            elif len(train_edges_false) == len(train_edges) and \
                            len(test_edges_false) == len(test_edges) and \
                            len(val_edges_false) == len(val_edges):
                break

    # Randomly generate false edges (idx_i, idx_j) 1 at a time to save memory
    elif false_edge_sampling == 'random':
        if verbose == True:
            print('creating false test edges...')

        # FALSE TEST EDGES
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:  # no self-loops
                continue

            # Ensure both endpoints are in largest WCC
            if idx_i not in largest_wcc_set or idx_j not in largest_wcc_set:
                continue

            false_edge = (idx_i, idx_j)

            # Make sure false_edge not an actual edge, and not a repeat
            if false_edge in all_edge_set:
                continue
            if false_edge in test_edges_false:
                continue

            test_edges_false.add(false_edge)

            if len(test_edges_false) % 100000 == 0 and verbose == True:
                print('Current num false test edges: ', len(test_edges_false))

        # FALSE VAL EDGES
        if verbose == True:
            print('creating false val edges...')

        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue

            false_edge = (idx_i, idx_j)

            # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
            if false_edge in all_edge_set or \
                            false_edge in test_edges_false or \
                            false_edge in val_edges_false:
                continue

            val_edges_false.add(false_edge)

            if len(val_edges_false) % 100000 == 0 and verbose == True:
                print('Current num false val edges: ', len(val_edges_false))

        # FALSE TRAIN EDGES
        if verbose == True:
            print('creating false train edges...')

        while len(train_edges_false) < len(train_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue

            false_edge = (idx_i, idx_j)

            # Make sure false_edge in not an actual edge, not in test_edges_false,
            # not in val_edges_false, not a repeat
            if false_edge in all_edge_set or \
                            false_edge in test_edges_false or \
                            false_edge in val_edges_false or \
                            false_edge in train_edges_false:
                continue

            train_edges_false.add(false_edge)

            if len(train_edges_false) % 100000 == 0 and verbose == True:
                print('Current num false train edges: ', len(train_edges_false))

    ### ---------- FINAL DISJOINTNESS CHECKS ---------- ###
    if verbose == True:
        print('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_set)
    assert val_edges_false.isdisjoint(all_edge_set)
    assert train_edges_false.isdisjoint(all_edge_set)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose == True:
        print('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print('Done with train-test split!')
        print('Num train edges (true, false): (', train_edges.shape[0], ', ', train_edges_false.shape[0], ')')
        print('Num test edges (true, false): (', test_edges.shape[0], ', ', test_edges_false.shape[0], ')')
        print('Num val edges (true, false): (', val_edges.shape[0], ', ', val_edges_false.shape[0], ')')
        print('')

    # Return final edge lists (edges can go either direction!)
    return adj_train, train_edges, train_edges_false, \
           val_edges, val_edges_false, test_edges, test_edges_false




class MLP(nn.Module):
    def __init__(self, emb_matrix,hidden_layer, dropout):
        super(MLP, self).__init__()
        """
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.emb_matrix = emb_matrix

        MLP_modules = []
        self.num_layers = len(hidden_layer)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                # MLP_modules.append(nn.Linear(np.shape(emb_matrix)[1], hidden_layer[0]))
                MLP_modules.append(nn.Linear(2, hidden_layer[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer[-1], 1)

        self.weight_vec = torch.empty(1,np.shape(emb_matrix)[1])

        self.weight_vec=nn.init.normal_(self.weight_vec)

        # print('Z******',self.weight_vec)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, edge,emb_matrix0,device):
        # print(emb_matrix0)
        emb1= emb_matrix0[edge[0],:]
        emb2= emb_matrix0[edge[1],:]
        # print(emb1)
        # print('####',emb1.size(), self.weight_vec.size())
        # print((self.weight_vec.to(device) * emb1).size(),(self.weight_vec.to(device) * emb2))
        embed_sim1 = torch.cosine_similarity((self.weight_vec.to(device)*emb1),(self.weight_vec.to(device)*emb2),dim=1)
        embed_sim2 = torch.dot((self.weight_vec.to(device) * emb1)[0],(self.weight_vec.to(device) * emb2)[0])
        # print('****',embed_sim1, embed_sim2)
        # print(embed_sim1.size(), embed_sim2.size())
        # interaction = torch.cat((embed_sim1[0], embed_sim2), 0)
        interaction = [embed_sim1[0], embed_sim2]
        interaction=torch.FloatTensor(interaction).to(device)
        output = self.MLP_layers(interaction)

        prediction = self.predict_layer(output)
        return prediction.view(-1)


class MLP_(nn.Module):
    def __init__(self, hidden_layer,dropout):
        super(MLP_, self).__init__()
        """
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        MLP_modules = []
        self.num_layers = len(hidden_layer)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                # MLP_modules.append(nn.Linear(np.shape(emb_matrix)[1], hidden_layer[0]))
                MLP_modules.append(nn.Linear( 3, hidden_layer[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer[-1], 1)

        # self.weight_vec = torch.empty(1,np.shape(emb_matrix)[1])
        #
        # self.weight_vec=nn.init.normal_(self.weight_vec)

        # print('Z******',self.weight_vec)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, X_train,device):

        output = self.MLP_layers(X_train)

        prediction = self.predict_layer(output)
        return prediction.view(-1)






class MLP_mia(nn.Module):
    def __init__(self, emb_matrix,hidden_layer, lay_1_dim,dropout):
        super(MLP_mia, self).__init__()
        """
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.emb_matrix = emb_matrix

        MLP_modules = []
        self.num_layers = len(hidden_layer)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                # MLP_modules.append(nn.Linear(np.shape(emb_matrix)[1], hidden_layer[0]))
                MLP_modules.append(nn.Linear( 2, hidden_layer[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer[-1], 1)

        self.weight_vec = torch.empty(1,np.shape(emb_matrix)[1])

        self.weight_vec=nn.init.normal_(self.weight_vec)

        # print('Z******',self.weight_vec)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, edge,emb_matrix0,device):
        print(edge.size())
        emb1= emb_matrix0[edge[:,0],:]
        emb2= emb_matrix0[edge[:,1],:]
        # print(emb1)
        # print('####',emb1.size(), self.weight_vec.size())
        print((self.weight_vec.to(device) * emb1).size(),(self.weight_vec.to(device) * emb2))
        embed_sim1 = torch.cosine_similarity((self.weight_vec.to(device)*emb1),(self.weight_vec.to(device)*emb2),dim=1)
        embed_sim2 = torch.mul((self.weight_vec.to(device) * emb1),(self.weight_vec.to(device) * emb2))
        print(embed_sim1.size(), embed_sim2.size())
        embed_sim2 = torch.sum(embed_sim2,dim=1)
        # print('****',embed_sim1, embed_sim2)
        print('@@@@@',embed_sim1.size(), embed_sim2.size())
        print('%%%%', embed_sim1.t().size(), embed_sim2.t().size())
        interaction = torch.cat((embed_sim1.reshape(-1,1), embed_sim2.reshape(-1,1)), 1)
        # interaction = [embed_sim1.t(), embed_sim2.t()]
        print(interaction.size())
        # interaction=torch.FloatTensor(interaction).to(device)
        output = self.MLP_layers(interaction)

        prediction = self.predict_layer(output)
        return prediction.view(-1)



class MLP_mia_white(nn.Module):
    def __init__(self, z1_trains,z2_trains,hidden_layer, lay_1_dim,dropout):
        super(MLP_mia_white, self).__init__()
        """
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.emb_matrix1 =  z1_trains
        self.emb_matrix2 = z2_trains

        MLP_modules = []
        self.num_layers = len(hidden_layer)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                # MLP_modules.append(nn.Linear(np.shape(emb_matrix)[1], hidden_layer[0]))
                MLP_modules.append(nn.Linear( 2*(lay_1_dim), hidden_layer[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer[-1], 1)

        self.weight_vec1 = torch.empty(1,np.shape(z1_trains)[2])

        self.weight_vec1=nn.init.normal_(self.weight_vec1)

        self.weight_vec2 = torch.empty(1,np.shape(z2_trains)[2])

        self.weight_vec2=nn.init.normal_(self.weight_vec2)

        # print('Z******',self.weight_vec)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, edge_list, z1_trains,z2_trains,device):
        # print(z1_trains.size())
        #
        # exit()

        embs_1 = []
        embs_2 = []

        edge=edge_list

        node1 = (edge[0]).int()
        node2 = (edge[1]).int()
        emb_1 = []
        for emb_matrix in z1_trains:
            emb1_1 = emb_matrix[node1]
            # print(np.shape(emb1))
            emb2_1 =emb_matrix[node2]

            # print(self.weight_vec1.size(),emb1_1.size())

            embed_sim1_1 = torch.cosine_similarity((self.weight_vec1.to(device) * emb1_1),
                                                   (self.weight_vec1.to(device) * emb2_1), dim=1)
            embed_sim2_1 = torch.mul((self.weight_vec1.to(device) * emb1_1), (self.weight_vec1.to(device) * emb2_1))

            embed_sim2_1 = torch.sum(embed_sim2_1, dim=1)

            emb_1.append(embed_sim1_1)
            emb_1.append(embed_sim2_1)

        embs_1.append(emb_1)

        emb_2 = []
        for emb_matrix in z2_trains:
            emb1_2= emb_matrix[node1]
            # print(np.shape(emb1))
            emb2_2 = emb_matrix[node2]

            embed_sim1_2 = torch.cosine_similarity((self.weight_vec2.to(device) * emb1_2),
                                                   (self.weight_vec2.to(device) * emb2_2), dim=1)
            embed_sim2_2 = torch.mul((self.weight_vec2.to(device) * emb1_2), (self.weight_vec2.to(device) * emb2_2))

            embed_sim2_2 =torch.sum(embed_sim2_2,dim=1)
            emb_2.append(embed_sim1_2.cpu().detach().numpy())
            emb_2.append(embed_sim2_2.cpu().detach().numpy())

        embs_2.append(emb_2)

        # print(embs_1.shape())
        sim_emb_1 = torch.FloatTensor(embs_1).to(device)
        sim_emb_2 = torch.FloatTensor(embs_2).to(device)

        # print(sim_emb_1.shape())
        #
        # exit()

        sim=sim_emb_1+sim_emb_2


        print('@@@@@',sim_emb_1.size(), sim_emb_2.size(),sim.size())
        # print('%%%%', embed_sim1_1.t().size(), embed_sim2_1.t().size())

        output = self.MLP_layers(sim)

        prediction = self.predict_layer(output)
        return prediction.view(-1)


class MLP_mia_white2(nn.Module):
    def __init__(self, z1_trains,z2_trains,hidden_layer, lay_1_dim,dropout):
        super(MLP_mia_white2, self).__init__()
        """
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.emb_matrix1 =  z1_trains
        self.emb_matrix2 = z2_trains

        MLP_modules = []
        self.num_layers = len(hidden_layer)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                # MLP_modules.append(nn.Linear(np.shape(emb_matrix)[1], hidden_layer[0]))
                MLP_modules.append(nn.Linear( 2*(lay_1_dim-1), hidden_layer[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer[-1], 1)

        self.weight_vec1 = torch.empty(1,np.shape(z1_trains)[2])

        self.weight_vec1=nn.init.normal_(self.weight_vec1)

        self.weight_vec2 = torch.empty(1,np.shape(z2_trains)[2])

        self.weight_vec2=nn.init.normal_(self.weight_vec2)

        self.weight_vec1 = torch.nn.Parameter(self.weight_vec1, requires_grad=True)
        self.weight_vec2 = torch.nn.Parameter(self.weight_vec2, requires_grad=True)

        # print('Z******',self.weight_vec)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, edge_list, z1_trains,z2_trains,device):
        print(z1_trains.size())
        #
        # exit()

        embs_1 = []
        embs_2 = []
        for edge in edge_list:
            node1 = (edge[0]).int()
            node2 = (edge[1]).int()
            emb_1 = []

            emb1_1 = z1_trains[:,node1,:]
            emb1_2 = z2_trains[:, node1, :]

            emb2_1 = z1_trains[:, node2, :]
            emb2_2 = z2_trains[:, node2, :]

            # print(emb1_1 )
            #
            # print(emb1_1.size())

            # print(self.weight_vec1.size(),emb1_1.size())
            # print(torch.mul(self.weight_vec1.to(device),emb1_1))

            embed_sim1_1 = torch.cosine_similarity(torch.mul(self.weight_vec1.to(device),emb1_1),
                                                   torch.mul(self.weight_vec1.to(device),emb2_1), dim=1)
            embed_sim2_1 = torch.mul(torch.mul(self.weight_vec1.to(device),emb1_1), torch.mul(self.weight_vec1.to(device),emb2_1))

            embed_sim2_1 = torch.sum(embed_sim2_1, dim=1)

            # print(embed_sim1_1.size(),embed_sim2_1.size())

            embed_sim1=(torch.cat((embed_sim1_1.reshape(-1,1),embed_sim2_1.reshape(-1,1)),dim=1)).reshape(-1)

            embed_sim1_2 = torch.cosine_similarity(torch.mul(self.weight_vec2.to(device), emb1_2),
                                                   torch.mul(self.weight_vec2.to(device), emb2_2), dim=1)
            embed_sim2_2 = torch.mul(torch.mul(self.weight_vec2.to(device), emb1_2),
                                     torch.mul(self.weight_vec2.to(device), emb2_2))

            embed_sim2_2 = torch.sum(embed_sim2_2, dim=1)

            embed_sim2 = (torch.cat((embed_sim1_2.reshape(-1,1), embed_sim2_2.reshape(-1,1)), dim=1)).reshape(-1)

            # embs_1.append(embed_sim1)
            #
            # embs_2.append(embed_sim2)



            # embs_1.append(embed_sim1_1)
            #
            # embs_2.append(embed_sim1_2)

            embs_1.append(embed_sim1.cpu().detach().numpy())

            embs_2.append(embed_sim2.cpu().detach().numpy())

        # embs_1=np.array(embs_1)
        # embs_2 = np.array(embs_2)
        # print(embs_1.shape())

        print(np.shape(embs_1),np.shape(embs_2))

        sim_emb_1 = torch.FloatTensor(embs_1).to(device)
        sim_emb_2 = torch.FloatTensor(embs_2).to(device)

        sim=sim_emb_1+sim_emb_2


        print('@@@@@',sim_emb_1.size(), sim_emb_2.size(),sim.size())
        # print('%%%%', embed_sim1_1.t().size(), embed_sim2_1.t().size())

        output = self.MLP_layers(sim)

        prediction = self.predict_layer(output)
        return prediction.view(-1)


class MLP_mia_white2_m(nn.Module):
    def  __init__(self,z1_trains,z2_trains,hidden_layer, lay_1_dim,dropout,ms):
        super(MLP_mia_white2_m, self).__init__()
        """
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.emb_matrix1 =  z1_trains
        self.emb_matrix2 = z2_trains

        MLP_modules = []
        self.num_layers = len(hidden_layer)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                # MLP_modules.append(nn.Linear(np.shape(emb_matrix)[1], hidden_layer[0]))
                # MLP_modules.append(nn.Linear( 2*(lay_1_dim-1), hidden_layer[0]))
                MLP_modules.append(nn.Linear((lay_1_dim - 1), hidden_layer[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer[-1], 1)

        self.weight_vec1 = torch.empty(ms,np.shape(z1_trains)[2])

        self.weight_vec1=nn.init.normal_(self.weight_vec1)

        self.weight_vec2 = torch.empty(ms,np.shape(z2_trains)[2])

        self.weight_vec2=nn.init.normal_(self.weight_vec2)

        self.weight_vec1 = torch.nn.Parameter(self.weight_vec1, requires_grad=True)
        self.weight_vec2 = torch.nn.Parameter(self.weight_vec2, requires_grad=True)

        # print('Z******',self.weight_vec)

        self.ms=ms

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, edge_list, z1_trains,z2_trains,device):
        print(z1_trains.size())
        #
        # exit()

        embs_1 = []
        embs_2 = []
        i=0
        for edge in edge_list:
            i+=1
            node1 = (edge[0]).int()
            node2 = (edge[1]).int()
            emb_1 = []

            emb1_1 = z1_trains[:,node1,:]
            emb1_2 = z2_trains[:, node1, :]

            emb2_1 = z1_trains[:, node2, :]
            emb2_2 = z2_trains[:, node2, :]

            # print(emb1_1 )
            #
            # print(emb1_1.size())

            # print(self.weight_vec1.size(),emb1_1.size())
            # print(torch.mul(self.weight_vec1.to(device),emb1_1))
            for m_ in range(self.ms):
                embed_sim1_1 = torch.cosine_similarity(torch.mul(self.weight_vec1[m_].to(device), emb1_1),
                                                       torch.mul(self.weight_vec1[m_].to(device), emb2_1), dim=1)
                # embed_sim2_1 = torch.mul(torch.mul(self.weight_vec1[m_].to(device), emb1_1),
                #                          torch.mul(self.weight_vec1[m_].to(device), emb2_1))
                #
                # embed_sim2_1 = torch.sum(embed_sim2_1, dim=1)
                #
                # # print(embed_sim1_1.size(),embed_sim2_1.size())
                #
                # embed_sim1 = (torch.cat((embed_sim1_1.reshape(-1, 1), embed_sim2_1.reshape(-1, 1)), dim=1)).reshape(-1)

                embed_sim1_2 = torch.cosine_similarity(torch.mul(self.weight_vec2[m_].to(device), emb1_2),
                                                       torch.mul(self.weight_vec2[m_].to(device), emb2_2), dim=1)
                # embed_sim2_2 = torch.mul(torch.mul(self.weight_vec2[m_].to(device), emb1_2),
                #                          torch.mul(self.weight_vec2[m_].to(device), emb2_2))
                #
                # embed_sim2_2 = torch.sum(embed_sim2_2, dim=1)
                #
                # embed_sim2 = (torch.cat((embed_sim1_2.reshape(-1, 1), embed_sim2_2.reshape(-1, 1)), dim=1)).reshape(-1)

                embed_sim1_1 = torch.unsqueeze(embed_sim1_1, dim=1)
                embed_sim1_2 = torch.unsqueeze(embed_sim1_2, dim=1)

                if m_==0:

                    embed_sim1_1_ = embed_sim1_1/self.ms
                    embed_sim1_2_ = embed_sim1_2/self.ms
                else:
                    embed_sim1_1_ += embed_sim1_1/self.ms
                    embed_sim1_2_+= embed_sim1_2/self.ms

            if i == 1:
                embs_1 = embed_sim1_1_
                embs_2 = embed_sim1_2_
            else:
                # print((embs_1).reshape(-1,1).size())
                embs_1 = torch.cat((embs_1, embed_sim1_1_), dim=1)
                embs_2 = torch.cat((embs_2, embed_sim1_2_), dim=1)


        sim_emb_1 = embs_1.t()
        sim_emb_2 = embs_2.t()

        sim = sim_emb_1 + sim_emb_2

        print('@@@@@', sim_emb_1.size(), sim_emb_2.size(), sim.size())
        # print('%%%%', embed_sim1_1.t().size(), embed_sim2_1.t().size())


        # output = self.MLP_layers(sim)

        output = self.MLP_layers(sim_emb_1)

        prediction = self.predict_layer(output)
        return prediction.view(-1)


class MLP_mia_white_weight_sim(nn.Module):
    def __init__(self, z1_trains,z2_trains,hidden_layer, lay_1_dim,dropout):
        super(MLP_mia_white_weight_sim, self).__init__()
        """
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.emb_matrix1 =  z1_trains
        self.emb_matrix2 = z2_trains

        MLP_modules = []
        self.num_layers = len(hidden_layer)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                # MLP_modules.append(nn.Linear(np.shape(emb_matrix)[1], hidden_layer[0]))
                MLP_modules.append(nn.Linear( 2*(lay_1_dim), hidden_layer[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer[-1], 1)

        self.weight_vec1 = torch.empty(1,2*np.shape(z1_trains)[0])

        self.weight_vec1=nn.init.normal_(self.weight_vec1)

        self.weight_vec2 = torch.empty(1,2*np.shape(z2_trains)[0])

        self.weight_vec2=nn.init.normal_(self.weight_vec2)

        # print('Z******',self.weight_vec)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, edge_list, z1_trains,z2_trains,device):
        # print(z1_trains.size())
        #
        # exit()

        embs_1 = []
        embs_2 = []

        edge=edge_list

        node1 = (edge[0]).int()
        node2 = (edge[1]).int()
        emb_1 = []
        for emb_matrix in z1_trains:
            emb1_1 = emb_matrix[node1]
            # print(np.shape(emb1))
            emb2_1 =emb_matrix[node2]

            # print(emb1_1.size(),emb1_1.size())

            # embed_sim1_1 = torch.cosine_similarity((self.weight_vec1.to(device) * emb1_1),
            #                                        (self.weight_vec1.to(device) * emb2_1), dim=1)
            # embed_sim2_1 = torch.mul((self.weight_vec1.to(device) * emb1_1), (self.weight_vec1.to(device) * emb2_1))
            #
            # embed_sim2_1 = torch.sum(embed_sim2_1, dim=1)

            emb1_1=emb1_1.unsqueeze(0).float()
            emb2_1=emb2_1.unsqueeze(0).float()

            embed_sim1_1 = torch.cosine_similarity(emb1_1,emb2_1, dim=1)
            embed_sim2_1 = torch.mul(emb1_1,emb2_1)

            embed_sim2_1 = torch.sum(embed_sim2_1, dim=1)

            emb_1.append(embed_sim1_1.cpu().detach().numpy()[0])
            emb_1.append(embed_sim2_1.cpu().detach().numpy()[0])

        embs_1=(emb_1)

        emb_2 = []
        for emb_matrix in z2_trains:
            emb1_2= emb_matrix[node1]
            # print(np.shape(emb1))
            emb2_2 = emb_matrix[node2]

            # embed_sim1_2 = torch.cosine_similarity((self.weight_vec2.to(device) * emb1_2),
            #                                        (self.weight_vec2.to(device) * emb2_2), dim=1)
            # embed_sim2_2 = torch.mul((self.weight_vec2.to(device) * emb1_2), (self.weight_vec2.to(device) * emb2_2))
            #
            # embed_sim2_2 =torch.sum(embed_sim2_2,dim=1)
            # emb_2.append(embed_sim1_2.cpu().detach().numpy())
            # emb_2.append(embed_sim2_2.cpu().detach().numpy())
            emb1_2 = emb1_2.unsqueeze(0).float()
            emb2_2 = emb2_2.unsqueeze(0).float()

            embed_sim1_2 = torch.cosine_similarity(emb1_2,emb2_2, dim=1)
            embed_sim2_2 = torch.mul(emb1_2,emb2_2)

            embed_sim2_2 = torch.sum(embed_sim2_2, dim=1)
            emb_2.append(embed_sim1_2.cpu().detach().numpy()[0])
            emb_2.append(embed_sim2_2.cpu().detach().numpy()[0])
            # print(embed_sim1_2)

        embs_2=(emb_2)
        # print(embs_2)

        embs_1=torch.FloatTensor(np.array(embs_1)).to(device)
        embs_2 = torch.FloatTensor(np.array(embs_2)).to(device)

        # print(embs_1.size(),embs_2.size())

        sim_emb_1 = (self.weight_vec1.to(device) * embs_1)
        sim_emb_2 = (self.weight_vec2.to(device) * embs_2)

        # print(embs_1.shape())
        # sim_emb_1 = torch.FloatTensor(self.weight_vec1.to(device) *embs_1).to(device)
        # sim_emb_2 = torch.FloatTensor(self.weight_vec2.to(device) *embs_2).to(device)

        # print(sim_emb_1.shape())
        #
        # exit()

        sim=sim_emb_1+sim_emb_2


        print('@@@@@',sim_emb_1.size(), sim_emb_2.size(),sim.size())
        # print('%%%%', embed_sim1_1.t().size(), embed_sim2_1.t().size())

        output = self.MLP_layers(sim)

        prediction = self.predict_layer(output)
        prediction = F.sigmoid(prediction)
        return prediction.view(-1)



class MLP_mia_white_weight_sim2(nn.Module):
    def __init__(self, z1_trains,z2_trains,hidden_layer, lay_1_dim,dropout):
        super(MLP_mia_white_weight_sim2, self).__init__()
        """
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.emb_matrix1 =  z1_trains
        self.emb_matrix2 = z2_trains

        MLP_modules = []
        self.num_layers = len(hidden_layer)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                # MLP_modules.append(nn.Linear(np.shape(emb_matrix)[1], hidden_layer[0]))
                MLP_modules.append(nn.Linear(lay_1_dim, hidden_layer[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer[-1], 1)


        weight_vec1 = torch.empty(1,np.shape(z1_trains)[0])

        weight_vec1=nn.init.normal_(weight_vec1)

        weight_vec2 = torch.empty(1,np.shape(z1_trains)[0])

        weight_vec2=nn.init.normal_(weight_vec2)

        # self.weight_vec1 = torch.autograd.Variable(weight_vec1, requires_grad=True)
        # self.weight_vec2 = torch.autograd.Variable(weight_vec2, requires_grad=True)

        self.weight_vec1 = torch.nn.Parameter(weight_vec1, requires_grad=True)
        self.weight_vec2 = torch.nn.Parameter(weight_vec2, requires_grad=True)

        # print('Z******',self.weight_vec)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, edge_list, z1_trains,z2_trains,device):
        print(z1_trains.size())
        #
        # exit()

        embs_1 =torch.Tensor().to(device)
        embs_2 =torch.Tensor().to(device)
        i=0
        for edge in edge_list:
            i+=1
            node1 = (edge[0]).int()
            node2 = (edge[1]).int()
            emb_1 = []

            emb1_1 = z1_trains[:,node1,:]
            emb1_2 = z2_trains[:, node1, :]

            emb2_1 = z1_trains[:, node2, :]
            emb2_2 = z2_trains[:, node2, :]

            # print(emb1_1 )
            #
            # print(emb1_1.size())

            # print(self.weight_vec1.size(),emb1_1.size())
            # print(torch.mul(self.weight_vec1.to(device),emb1_1))

            embed_sim1_1 = torch.cosine_similarity(emb1_1,emb2_1, dim=1)
            embed_sim2_1 = torch.mul(emb1_1, emb2_1)

            embed_sim2_1 = torch.sum(embed_sim2_1, dim=1)

            # print(embed_sim1_1.size(),embed_sim2_1.size())

            embed_sim1=(torch.cat((embed_sim1_1.reshape(-1,1),embed_sim2_1.reshape(-1,1)),dim=1)).reshape(-1)

            embed_sim1_2 = torch.cosine_similarity( emb1_2, emb2_2, dim=1)
            embed_sim2_2 = torch.mul(emb1_2,emb2_2)

            embed_sim2_2 = torch.sum(embed_sim2_2, dim=1)

            embed_sim2 = (torch.cat((embed_sim1_2.reshape(-1,1), embed_sim2_2.reshape(-1,1)), dim=1)).reshape(-1)

            # embs_1.append(embed_sim1)
            #
            # embs_2.append(embed_sim2)
            # print('###',i,embed_sim1_1.size())

            embed_sim1_1=torch.unsqueeze(embed_sim1_1,dim=1)
            embed_sim1_2=torch.unsqueeze(embed_sim1_2,dim=1)

            if i==1:
                embs_1 = embed_sim1_1
                embs_2 = embed_sim1_2
            else:
                # print((embs_1).reshape(-1,1).size())
                embs_1=torch.cat((embs_1,embed_sim1_1),dim=1)
                embs_2 = torch.cat((embs_2,embed_sim1_2),dim=1)

                # print(embs_1.size(),embs_2.size())

            # print('###', i,embs_1.size())

            # embs_1.append(embed_sim1.cpu().detach().numpy())
            #
            # embs_2.append(embed_sim2.cpu().detach().numpy())

        # embs_1=np.array(embs_1)
        # embs_2 = np.array(embs_2)
        # print(embs_1.shape())

        # print(np.shape(embs_1),np.shape(embs_2))

        embs_1=embs_1.t()
        embs_2 = embs_2.t()

        # embs_1=torch.FloatTensor(np.array(embs_1)).to(device)
        # embs_2 = torch.FloatTensor(np.array(embs_2)).to(device)

        # print(embs_1.size(),embs_2.size())

        sim_emb_1 = (self.weight_vec1.to(device) * embs_1)
        sim_emb_2 = (self.weight_vec2.to(device) * embs_2)

        # sim_emb_1 = torch.FloatTensor(embs_1).to(device)
        # sim_emb_2 = torch.FloatTensor(embs_2).to(device)

        # print('&&&',self.weight_vec1,self.weight_vec2)

        sim=sim_emb_1+sim_emb_2


        # print('@@@@@',sim_emb_1.size(), sim_emb_2.size(),sim.size())
        # print('%%%%', embed_sim1_1.t().size(), embed_sim2_1.t().size())

        # print(self.MLP_layers)

        output = self.MLP_layers(sim)

        prediction = self.predict_layer(output)
        # prediction = F.sigmoid(prediction)
        return prediction.view(-1)


class MLP_mia_white_weight_sim2_m(nn.Module):
    def __init__(self, z1_trains,z2_trains,hidden_layer, lay_1_dim,dropout,ms):
        super(MLP_mia_white_weight_sim2_m, self).__init__()
        """
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.emb_matrix1 =  z1_trains
        self.emb_matrix2 = z2_trains

        MLP_modules = []
        self.num_layers = len(hidden_layer)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                # MLP_modules.append(nn.Linear(np.shape(emb_matrix)[1], hidden_layer[0]))
                MLP_modules.append(nn.Linear(lay_1_dim, hidden_layer[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer[-1], 1)


        weight_vec1 = torch.empty(ms,np.shape(z1_trains)[0])

        weight_vec1=nn.init.normal_(weight_vec1)

        weight_vec2 = torch.empty(ms,np.shape(z1_trains)[0])

        weight_vec2=nn.init.normal_(weight_vec2)

        # self.weight_vec1 = torch.autograd.Variable(weight_vec1, requires_grad=True)
        # self.weight_vec2 = torch.autograd.Variable(weight_vec2, requires_grad=True)

        self.weight_vec1 = torch.nn.Parameter(weight_vec1, requires_grad=True)
        self.weight_vec2 = torch.nn.Parameter(weight_vec2, requires_grad=True)

        self.ms = ms

        # print('Z******',self.weight_vec)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, edge_list, z1_trains,z2_trains,device):
        print(z1_trains.size())
        #
        # exit()

        embs_1 =torch.Tensor().to(device)
        embs_2 =torch.Tensor().to(device)
        i=0
        for edge in edge_list:
            i+=1
            node1 = (edge[0]).int()
            node2 = (edge[1]).int()
            emb_1 = []

            emb1_1 = z1_trains[:,node1,:]
            emb1_2 = z2_trains[:, node1, :]

            emb2_1 = z1_trains[:, node2, :]
            emb2_2 = z2_trains[:, node2, :]

            # print(emb1_1 )
            #
            # print(emb1_1.size())

            # print(self.weight_vec1.size(),emb1_1.size())
            # print(torch.mul(self.weight_vec1.to(device),emb1_1))

            embed_sim1_1 = torch.cosine_similarity(emb1_1,emb2_1, dim=1)
            embed_sim2_1 = torch.mul(emb1_1, emb2_1)

            embed_sim2_1 = torch.sum(embed_sim2_1, dim=1)

            # print(embed_sim1_1.size(),embed_sim2_1.size())

            embed_sim1=(torch.cat((embed_sim1_1.reshape(-1,1),embed_sim2_1.reshape(-1,1)),dim=1)).reshape(-1)

            embed_sim1_2 = torch.cosine_similarity( emb1_2, emb2_2, dim=1)
            embed_sim2_2 = torch.mul(emb1_2,emb2_2)

            embed_sim2_2 = torch.sum(embed_sim2_2, dim=1)

            embed_sim2 = (torch.cat((embed_sim1_2.reshape(-1,1), embed_sim2_2.reshape(-1,1)), dim=1)).reshape(-1)

            # embs_1.append(embed_sim1)
            #
            # embs_2.append(embed_sim2)
            # print('###',i,embed_sim1_1.size())

            embed_sim1_1=torch.unsqueeze(embed_sim1_1,dim=1)
            embed_sim1_2=torch.unsqueeze(embed_sim1_2,dim=1)

            if i==1:
                embs_1 = embed_sim1_1
                embs_2 = embed_sim1_2
            else:
                # print((embs_1).reshape(-1,1).size())
                embs_1=torch.cat((embs_1,embed_sim1_1),dim=1)
                embs_2 = torch.cat((embs_2,embed_sim1_2),dim=1)

                # print(embs_1.size(),embs_2.size())

            # print('###', i,embs_1.size())

            # embs_1.append(embed_sim1.cpu().detach().numpy())
            #
            # embs_2.append(embed_sim2.cpu().detach().numpy())

        # embs_1=np.array(embs_1)
        # embs_2 = np.array(embs_2)
        # print(embs_1.shape())

        # print(np.shape(embs_1),np.shape(embs_2))

        embs_1=embs_1.t()
        embs_2 = embs_2.t()

        # embs_1=torch.FloatTensor(np.array(embs_1)).to(device)
        # embs_2 = torch.FloatTensor(np.array(embs_2)).to(device)

        # print(embs_1.size(),embs_2.size())

        for m_ in range(self.ms):

            sim_emb_1 = (self.weight_vec1[m_].to(device) * embs_1)
            sim_emb_2 = (self.weight_vec2[m_].to(device) * embs_2)

            # sim_emb_1 = torch.FloatTensor(embs_1).to(device)
            # sim_emb_2 = torch.FloatTensor(embs_2).to(device)

            # print('&&&',self.weight_vec1,self.weight_vec2)

            # print(sim_emb_1.size())

            if m_==0:
                sim_emb_1_ =sim_emb_1/self.ms
                sim_emb_2_ = sim_emb_2/self.ms

            else:
                sim_emb_1_ += sim_emb_1/self.ms
                sim_emb_2_ += sim_emb_2/self.ms

        # sim_emb_1=torch.sum(sim_emb_1)
        # sim_emb_2 = torch.sum(sim_emb_2)
        #
        # print( sim_emb_1.size())

        sim=sim_emb_1_+sim_emb_2_

        # exit()


        # print('@@@@@',sim_emb_1.size(), sim_emb_2.size(),sim.size())
        # print('%%%%', embed_sim1_1.t().size(), embed_sim2_1.t().size())

        # print(self.MLP_layers)

        output = self.MLP_layers(sim)

        prediction = self.predict_layer(output)
        # prediction = F.sigmoid(prediction)
        return prediction.view(-1)
