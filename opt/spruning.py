from .base import BaseOp
from netflow import *
from .utils import *

import copy
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from misc.train import train_model


class SPruningOp(BaseOp):
    def __init__(self, model: nn.Module, amount=0.8, method="l1"):
        super().__init__(model)
        self.op_name = "structured_pruning"
        self.amount = amount
        self.method = method
        self.config = None

    def apply(self, name_list, verbose=False, with_profile=False, *args, **kwargs):
        name_set = set()
        diff = {}
        storage_save = {}
        for name in name_list:
            if name not in self.operatable:
                print("{} is not a operatable layer, retry something in:{} !".format(name, self.operatable))
                raise AttributeError
            name_set.add(name)

        model_to_prune = copy.deepcopy(self.model)
        for mod_name, mod in model_to_prune.named_modules():
            if mod_name in name_set:
                if with_profile:
                    weight = mod.weight.data.cpu().numpy().flatten()
                    if hasattr(mod, "bias"):
                        bias = mod.bias.data.cpu().numpy().flatten()
                        param = np.concatenate([weight, bias], axis=0)
                    else:
                        param = weight
                if verbose:
                    print(f"Module weights before pruning: {list(mod.named_parameters())}")
                self._prune(mod)
                if with_profile:
                    weight = mod.weight.data.cpu().numpy().flatten()
                    if hasattr(mod, "bias"):
                        bias = mod.bias.data.cpu().numpy().flatten()
                        param_ = np.concatenate([weight, bias], axis=0)
                    else:
                        param_ = weight
                    diff[mod_name] = param - param_
                    storage_save[mod_name] = param.size * get_size(mod.weight.dtype) * self.amount
                if verbose:
                    print(f"Module weights after pruning: {list(mod.named_parameters())}")
        self.mod_model = model_to_prune
        if with_profile:
            return self.mod_model, diff, storage_save
        else:
            return self.mod_model

    def apply_with_finetune(self, name_list, verbose=False, *args, **kwargs):
        mod_model = self.apply(name_list, verbose)

        print("Finetuning...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mod_model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(mod_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.3)
        self.mod_model = train_model(mod_model, criterion, optimizer_ft, exp_lr_scheduler,
                                     num_epochs=2, device=device)
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
