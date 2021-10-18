from opt import *
import torch.nn as nn
from metric.metric_base import Metric
from netflow.net_interpreter import NetIntBase


class CookBase(object):
    def __init__(self, model: nn.Module, ops: List[BaseOp], metric: Metric, flow: NetIntBase, rate=0):
        self.model = model
        self.ops = ops
        self.metric = metric
        self.rate = rate
        self.flow = flow

    def run(self) -> nn.Module:
        raise NotImplementedError
