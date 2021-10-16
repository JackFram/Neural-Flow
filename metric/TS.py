from metric.metric_base import Metric
import torch
import numpy as np


class TopologySimilarity(Metric):
    def __init__(self):
        super().__init__()

    def get_topo(self, feature):
        B = feature.shape[0]  # get batch number
        feature = feature.view(B, -1)
        topo = torch.cdist(feature, feature, 2).cpu().numpy()[np.triu_indices(B - 1)]
        print(topo.shape)
        topo = topo - topo.mean()
        norm = np.linalg.norm(topo)
        topo = topo / norm
        return topo

    def get_batch_score(self, feature1, feature2):
        '''

        :param feature1: Torch.Tensor from model layer, shape (B, ...)
        :param feature2: Torch.Tensor from model layer, shape (B, ...)
        :return: Scalar, complexity level of the feature transformation

        '''
        norm_feature1_topo = self.get_topo(feature1)
        norm_feature2_topo = self.get_topo(feature2)
        TS_score = norm_feature1_topo.T @ norm_feature2_topo

        return TS_score

