from utils import *
import torch.nn as nn


class NetIntBase(object):
    def __init__(self, module:nn.Module):
        self.module = module
        self.graph = None

    def __len__(self):
        raise NotImplementedError

    def get_feature_list(self):
        raise NotImplementedError

    def get_name_list(self):
        raise NotImplementedError
