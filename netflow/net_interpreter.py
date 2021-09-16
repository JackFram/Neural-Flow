from utils import *
import torch.nn as nn


class NetIntBase(object):
    def __init__(self, net:nn.Module):
        self.idx_2_name = {}
        self.net = net
        self.seq_net = None

    def __getattr__(self, item: int):
        return self.seq_net[:item+1]

    def __len__(self):
        return len(self.idx_2_name)

    def sequentialize(self):
        '''

        :return: no return, instantiate self.seq_net
        '''

        raise NotImplementedError

