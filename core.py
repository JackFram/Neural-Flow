import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from approx_optim import *
from approx_optim.pruning import PruningOp
from misc.eval import eval

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
    op = QuantizeOp(model)
    op.set_config(name_list)
    print(op.operatable)

    metric = TopologySimilarity()
    for name in op.operatable:
        idx = name_list.index(name)
        print(name)
        print(metric.get_batch_score(feature_list[0], feature_list[idx]))

    op.apply(name_list[1:2])
    eval(op.mod_model, testloader)
    op.model

    # TODO: evaluation and visualization

