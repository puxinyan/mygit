import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
from torch.nn import _reduction as _Reduction
from torch._C import _infer_size, _add_docstr



def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sp.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalizemx(mx):
    degrees = mx.sum(axis=0)[0].tolist()
    #    print(degrees)
    D = sp.diags(degrees, [0])
    D = D.power(-1)
    mx = mx.dot(D)
    return mx


def scattering1st(spmx, order):
    I_n = sp.eye(spmx.shape[0])
    adj_sct = 0.5 * (spmx + I_n)
    adj_power = adj_sct
    adj_power = sparse_mx_to_torch_sparse_tensor(adj_power).cuda()
    adj_sct = sparse_mx_to_torch_sparse_tensor(adj_sct).cuda()
    I_n = sparse_mx_to_torch_sparse_tensor(I_n)
    if order == 1:
        adj_int = torch.spmm((adj_power - I_n.cuda()), adj_power.to_dense())
    elif order < 16:
        for i in range(order - 1):
            adj_power = torch.spmm(adj_power, adj_sct.to_dense())
            print('Generating SCT')
        adj_int = torch.spmm((adj_power - I_n.cuda()), adj_power)
    elif order == 16:
        for i in range(order - 1):
            adj_power = torch.spmm(adj_power, adj_sct.to_dense())
            print('Generating SCT')
        adj_int = torch.spmm(adj_power, adj_power)
    return adj_int


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    #   take from https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
    #    idx_train = range(140)
    #    idx_val = range(200, 500)
    #    idx_test = range(500, 1500)

    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    features = normalize(features)
    A_tilde = normalize_adjacency_matrix(adj, sp.eye(adj.shape[0]))
    adj = normalizemx(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    print('Loading')
    adj_sct1 = scattering1st(adj, 1)  ## psi_1 = P(I-P)
    adj_sct2 = scattering1st(adj, 2)  # psi_2 = P^2(I-P^2)
    adj_sct4 = scattering1st(adj, 4)  # psi_3 = P^4(I-P^4)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    A_tilde = sparse_mx_to_torch_sparse_tensor(A_tilde)
    return adj, A_tilde, adj_sct1, adj_sct2, adj_sct4, features, labels, idx_train, idx_val, idx_test


def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter() - t
    return features, precompute_time


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
