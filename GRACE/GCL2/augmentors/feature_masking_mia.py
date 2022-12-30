from GCL2.augmentors.augmentor import Graph, Augmentor
from GCL2.augmentors.functional import drop_feature,drop_feature_mia


class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class FeatureMaskingMia(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMaskingMia, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x,drop_mask = drop_feature_mia(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights),drop_mask