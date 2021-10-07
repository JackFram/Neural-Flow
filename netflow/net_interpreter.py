from utils import *
import torch.nn as nn


class NetIntBase(object):
    def __init__(self, net:nn.Module):
        self.net = net
        self.graph = None

    def __getattr__(self, *kargs):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

