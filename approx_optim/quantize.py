import torch
import torch.nn as nn
from .base import BaseOp


class QuantizeOp(BaseOp):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def apply(self, name, *kargs):
