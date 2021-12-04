import numpy as np
import torch
import torch.nn as nn


class BaseSolver:
    def __init__(self, net, ops: list):
        self.net = net
        self.ops = ops

    def get_profile(self, layer_name):
        raise NotImplementedError

    def get_solution(self, storage_threshold):
        raise NotImplementedError

    @property
    def operatable(self):
        ret_list = []
        for name, mod in self.net.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
                ret_list.append(name)
        return ret_list