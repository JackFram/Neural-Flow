
from .base import BaseOp
from netflow import *

import copy
import torch.nn as nn
import torch.nn.utils.prune as prune

class PruningOp(BaseOp):
    def __init__(self, model: nn.Module, amount=0.9, method="random"):
        super().__init__(model)
        self.op_name = "pruning"
        self.amount = amount
        self.method = method
        # self.prunable()

    def apply(self, name):
        model_to_prune = copy.deepcopy(self.model)
        module_name = name.replace("_", ".")
        print(module_name)
        for name, mod in model_to_prune.named_modules():
            if name == module_name:
                print(list(mod.named_parameters()))
                self._prune(mod)
                print(list(mod.named_parameters()))
        self.mod_model = model_to_prune


    def _prune(self, module:nn.Module):
        if self.method == "l1":
            prune.l1_unstructured(module, 'weight', amount=self.amount)
        elif self.method == "random":
            prune.random_unstructured(module, 'weight', amount=self.amount)

        prune.remove(module, 'weight')


    @property
    def operatable(self):
        ret_list = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
                ret_list.append(name)
