import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from approx_optim import *

from misc.eval import eval
from model import *
from netflow import *
from metric import TopologySimilarity
from approx_optim import QuantizeOp


if __name__ == "__main__":
    model = ResNet18()
    flow = FxInt(model)
    flow.run(torch.randn(10, 3, 32, 32))
    feature_list = flow.get_feature_list()
    name_list = flow.get_name_list()
    metric = TopologySimilarity()
    print(metric.get_batch_score(feature_list[1], feature_list[3]))
    print(name_list)
    # TODO: add quantization package and use it here
    op = QuantizeOp(model)
    op.get_config()
    op.apply()
