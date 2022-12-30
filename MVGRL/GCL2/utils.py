from typing import *
import os
import torch
import dgl
import random
import numpy as np
from torch import nn


def split_dataset(dataset, split_mode, *args, **kwargs):
    assert split_mode in ['rand', 'ogb', 'wikics', 'preload']
    if split_mode == 'rand':
        assert 'train_ratio' in kwargs and 'test_ratio' in kwargs
        train_ratio = kwargs['train_ratio']
        test_ratio = kwargs['test_ratio']
        num_samples = dataset.x.size(0)
        train_size = int(num_samples * train_ratio)
        test_size = int(num_samples * test_ratio)
        indices = torch.randperm(num_samples)
        return {
            'train': indices[:train_size],
            'val': indices[train_size: test_size + train_size],
            'test': indices[test_size + train_size:]
        }
    elif split_mode == 'ogb':
        return dataset.get_idx_split()
    elif split_mode == 'wikics':
        assert 'split_idx' in kwargs
        split_idx = kwargs['split_idx']
        return {
            'train': dataset.train_mask[:, split_idx],
            'test': dataset.test_mask,
            'val': dataset.val_mask[:, split_idx]
        }
    elif split_mode == 'preload':
        assert 'preload_split' in kwargs
        assert kwargs['preload_split'] is not None
        train_mask, test_mask, val_mask = kwargs['preload_split']
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(s):
    return (s.max() - s) / (s.max() - s.mean())


def build_dgl_graph(edge_index: torch.Tensor) -> dgl.DGLGraph:
    row, col = edge_index
    return dgl.graph((row, col))


def batchify_dict(dicts: List[dict], aggr_func=lambda x: x):
    res = dict()
    for d in dicts:
        for k, v in d.items():
            if k not in res:
                res[k] = [v]
            else:
                res[k].append(v)
    res = {k: aggr_func(v) for k, v in res.items()}
    return res


class MLP(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, hidden_layer, dropout):
        super(MLP, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        embedding_dim: number of embedding dimensions;
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.embed_user = nn.Embedding(user_num, embedding_dim)
        self.embed_item = nn.Embedding(item_num, embedding_dim)

        MLP_modules = []
        self.num_layers = len(hidden_layer)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                MLP_modules.append(nn.Linear(embedding_dim*2, hidden_layer[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer[-1], 1)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        interaction = torch.cat((embed_user, embed_item), -1)
        output = self.MLP_layers(interaction)

        prediction = self.predict_layer(output)
        return prediction.view(-1)