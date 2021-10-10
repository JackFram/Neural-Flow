import numpy as np


class Metric(object):
    def __init__(self):
        pass

    def get_batch_score(self, feature1, feature2):
        '''

        :param feature1: Torch.Tensor from model layer, shape (B, ...)
        :param feature2: Torch.Tensor from model layer, shape (B, ...)
        :return: Scalar, complexity level of the feature transformation
        '''

        raise NotImplementedError

    def get_all_layer_batch_score(self, feature_list):
        '''

        :param feature_list: List[Torch.Tensor] with length N as layer number
        :return: Numpy Matrix with shape (N, N)
        '''
        N = len(feature_list)
        ret = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                ret[i, j] = self.get_batch_score(feature_list[i], feature_list[j])
        return ret