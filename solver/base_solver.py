import numpy as np
import torch
import torch.nn as nn


class BaseSolver:
    def __init__(self, net, ops: list):
        self.net = net
        self.ops = ops

    def get_profile(self, layer_name):
        raise NotImplementedError

    def get_solution(self, storage, latency):
        raise NotImplementedError