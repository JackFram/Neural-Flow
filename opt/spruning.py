from .base import BaseOp
# from netflow import *
# from .utils import *

import copy
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune
# import torch.optim as optim
# from misc.train import train_model


class SPruningOp(BaseOp):
    def __init__(self, model: nn.Module, amount=0.8, method="l1"):
        super().__init__(model)
        self.op_name = "spruning"
        self.amount = amount
        self.method = method
        self.config = None

    def apply(self, name_list, verbose=False, amount=None, with_profile=False, inplace=False, *args, **kwargs):
        if amount is not None:
            self.amount = amount
        diff = {}
        storage_save = {}
        if inplace is True:
            model_to_prune = self.model
        else:
            model_to_prune = copy.deepcopy(self.model)
        for name in set(name_list):
            if name + ".SVDLinear-0" in self.operatable:
                mod_1 = model_to_prune.get_submodule(name+".SVDLinear-0")
                mod_2 = model_to_prune.get_submodule(name+".SVDLinear-1")
                self._prune(mod_1)
                self._prune(mod_2)
                self.mod_model = model_to_prune
                if with_profile:
                    raise ValueError("Currently doesn't support get profile for SVD layer.")
                return self.mod_model

            if name + ".SVDConv-0" in self.operatable:
                mod_1 = model_to_prune.get_submodule(name+".SVDConv-0")
                mod_2 = model_to_prune.get_submodule(name+".SVDConv-1")
                self._prune(mod_1)
                self._prune(mod_2)
                self.mod_model = model_to_prune
                if with_profile:
                    raise ValueError("Currently doesn't support get profile for SVD layer.")
                return self.mod_model

            if name not in self.operatable:
                print("{} is not a operatable layer, retry something in:{} !".format(name, self.operatable))
                raise AttributeError

            mod = model_to_prune.get_submodule(name)
            if with_profile:
                param = self.get_param(mod)

            if verbose:
                print(f"Module weights before pruning: {list(mod.named_parameters())}")
            self._prune(mod)
            if with_profile:
                param_ = self.get_param(mod)
                diff[name] = param - param_
                # print(amount, np.linalg.norm(diff[name]), np.abs(diff[name]).max())
                storage_save[name] = 1 - self.amount
            if verbose:
                print(f"Module weights after pruning: {list(mod.named_parameters())}")

        self.mod_model = model_to_prune
        if with_profile:
            return self.mod_model, diff, storage_save
        else:
            return self.mod_model

    def _prune(self, module: nn.Module):
        if self.method == "l1":
            prune.ln_structured(
                module, 'weight', amount=self.amount, dim=1, n=1
            )
        elif self.method == "l2":
            prune.ln_structured(
                module, 'weight', amount=self.amount, dim=1, n=2
            )
        # prune.remove(module, 'weight')

    def set_config(self, config={}):
        self.config = config

    def get_param(self, mod:nn.modules):
        weight = mod.weight.data.cpu().numpy().flatten()
        if hasattr(mod, "bias") and mod.bias is not None:
            bias = mod.bias.data.cpu().numpy().flatten()
            return np.concatenate([weight, bias], axis=0)
        return weight

    def reset(self):
        self.config = None

    @property
    def operatable(self):
        ret_list = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
                ret_list.append(name)
        return ret_list
