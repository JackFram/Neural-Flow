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
    print(name_list)

    # metric = TopologySimilarity()
    # print(metric.get_batch_score(feature_list[1], feature_list[3]))

    op = QuantizeOp(model)
    print(op.quantizable)
    op.get_config(name_list=name_list)
    op.apply(verbose=False)

    # TODO: evaluation and visualization

