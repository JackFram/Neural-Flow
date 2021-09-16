from metric.metric_base import Metric
import torch
import numpy as np


class TopologySimilarity(Metric):
    def __init__(self):
        super().__init__()

    def get_score(self, feature1, feature2):
        '''

        :param feature1: Torch.Tensor from model layer, shape (B, ...)
        :param feature2: Torch.Tensor from model layer, shape (B, ...)
        :return: Scalar, complexity level of the feature transformation
        '''
        B = feature1.shape[0]
        feature1 = feature1.view(B, -1)
        feature2 = feature2.view(B, -1)

        # TODO: find shape and check no diagonal elements
        feature1_topo = torch.cdist(feature1, feature1, 2).numpy()[np.triu_indices(B - 1)]
        feature2_topo = torch.cdist(feature1, feature1, 2).numpy()[np.triu_indices(B - 1)]  # TODO: find shape

        norm_feature1 = (feature1 - feature1.mean()) / np.linalg.norm(feature1)
        norm_feature2 = (feature2 - feature2.mean()) / np.linalg.norm(feature2)

        return norm_feature1.T @ norm_feature2
