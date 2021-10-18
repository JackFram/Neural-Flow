from opt import *
import torch.nn as nn
from metric.metric_base import Metric
from netflow.net_interpreter import NetIntBase


class CookBase(object):
    def __init__(self, model: nn.Module, ops: [BaseOp], metric: Metric, flow: NetIntBase):
        self.model = model
        self.ops = ops
        self.metric = metric
        self.flow = flow

    def run(self, *args, **kwargs) -> nn.Module:
        raise NotImplementedError
