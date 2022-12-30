from GCL2.augmentors.augmentor import Graph, Augmentor
from GCL2.augmentors.functional import compute_ppr
import numpy as np

class PPRDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.2, eps: float = 1e-4, use_cache: bool = True, add_self_loop: bool = True):
        super(PPRDiffusion, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def augment(self, g: Graph) -> Graph:
        if self._cache is not None and self.use_cache:
            return self._cache
        x, edge_index, edge_weights = g.unfold()

        # print(edge_index)
        # print(np.shape(edge_index))
        # print('111')
        edge_index, edge_weights = compute_ppr(
            edge_index, edge_weights,
            alpha=self.alpha, eps=self.eps, ignore_edge_attr=False, add_self_loop=self.add_self_loop
        )
        # print(edge_index)
        # print(np.shape(edge_index))
        # print('222')
        res = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        self._cache = res
        return res
