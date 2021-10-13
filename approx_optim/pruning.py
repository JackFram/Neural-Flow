
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

    def apply(self, name_list, verbose=False, *args, **kwargs):
        name_set = set()
        for name in name_list:
            if name not in self.operatable:
                print("{} is not a operatable layer, retry something in:{} !".format(name, self.operatable))
                raise AttributeError
            name_set.add(name)
            
        model_to_prune = copy.deepcopy(self.model)
        for mod_name, mod in model_to_prune.named_modules():
            if mod_name in name_set:
                if verbose:
                    print(f"Module weights before pruning: {list(mod.named_parameters())}")
                self._prune(mod)
                if verbose:
                    print(f"Module weights after pruning: {list(mod.named_parameters())}")
        self.mod_model = model_to_prune

        return self.mod_model


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
        return ret_list