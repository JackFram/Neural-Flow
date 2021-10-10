import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from approx_optim import *
from misc.eval import eval
from model import *
from netflow import *


if __name__ == "__main__":
    model = ResNet18()
    flow = FxInt(model)
    flow.run(torch.randn(1, 3, 32, 32))
