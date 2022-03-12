from .base import BaseOp

import copy
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune

class SPruningOp(BaseOp):
    def __init__(self, model: nn.Module, method="l1"):
        super().__init__(model)
        self.op_name = "spruning"
        self.method = method
        self.config = None

    def apply(self, name_list, verbose=False, amount=None, remove_prune=False, inplace=False, *args, **kwargs):
        self.amount = amount
        self.remove_prune = remove_prune

        if inplace:
            model = self.model
        else:
            model = copy.deepcopy(self.model)

        for name in set(name_list):
            if name + ".SVDLinear-0" in self.operatable:
                mod_1 = model.get_submodule(name+".SVDLinear-0")
                mod_2 = model.get_submodule(name+".SVDLinear-1")
                self._prune(mod_1)
                self._prune(mod_2)
                self.mod_model = model
                return self.mod_model

            if name + ".SVDConv-0" in self.operatable:
                mod_1 = model.get_submodule(name+".SVDConv-0")
                mod_2 = model.get_submodule(name+".SVDConv-1")
                self._prune(mod_1)
                self._prune(mod_2)
                self.mod_model = model
                return self.mod_model

            if name not in self.operatable:
                print("{} is not a operatable layer, retry something in:{} !".format(name, self.operatable))
                raise AttributeError

            mod = model.get_submodule(name)

            if verbose:
                print(f"Module weights before pruning: {list(mod.named_parameters())}")
            self._prune(mod)
            if verbose:
                print(f"Module weights after pruning: {list(mod.named_parameters())}")

        self.mod_model = model
        return self.mod_model

    def _prune(self, module: nn.Module):
        if self.method == "l1":
            prune.ln_structured(module, 'weight', amount=self.amount, dim=1, n=1)
        elif self.method == "l2":
            prune.ln_structured(module, 'weight', amount=self.amount, dim=1, n=2)
        if self.remove_prune:
            prune.remove(module, 'weight')

    def set_config(self, config={}):
        self.config = config

    def reset(self):
        self.config = None

    @property
    def operatable(self):
        ret_list = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
                ret_list.append(name)
        return ret_list
