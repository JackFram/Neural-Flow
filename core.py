import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NNGraph:
    def __init__(self, net):
        self.model = net.eval()

    