import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseOp
from .utils import get_size
from collections import OrderedDict
# from misc.train import train_model

from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, get_default_qconfig


class BertQuantizeOp(BaseOp):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.op_name = "quantize"
        self.qconfig = None
        self.name_list = None
        self.qconfig_dict = {

        }
        self.qconfig_set = set()

    def apply(self, name_list: list=None, verbose=False, with_profile=False, inplace=False, *args, **kwargs):

        '''

        :param name_list:
        :param verbose:
        :param with_diff:
        :param args:
        :param kwargs:
        :return:
        '''
        if self.qconfig is None:
            if inplace is True:
                self.mod_model = self.model
            else:
                self.mod_model = copy.deepcopy(self.model)
            return self.mod_model

        if name_list is None:
            name_list = self.operatable
            self.qconfig_dict = {nn.Linear: self.qconfig}
        else:
            for name in name_list:
                if name + ".SVDLinear-0" in self.operatable:
                    self.qconfig_dict[name + ".SVDLinear-0"] = self.qconfig
                    self.qconfig_dict[name + ".SVDLinear-1"] = self.qconfig
                elif name + ".SVDConv-0" in self.operatable:
                    self.qconfig_dict[name + ".SVDConv-0"] = self.qconfig
                    self.qconfig_dict[name + ".SVDConv-1"] = self.qconfig
                elif name not in self.operatable:
                    print("{} is not a quantizable layer, retry something in:{} !".format(name, self.operatable))
                    raise AttributeError
                elif isinstance(self.qconfig, torch.dtype):
                    self.qconfig_set.add(self.qconfig)
                else:
                    self.qconfig_dict[name] = self.qconfig
        if isinstance(self.qconfig, torch.dtype):
            self.mod_model = torch.quantization.quantize_dynamic(
                self.model, self.qconfig_set, self.qconfig  # qint8, float16, quint8
            )
        else:
            self.mod_model = torch.quantization.quantize_dynamic(
                self.model, self.qconfig_dict  # qint8, float16, quint8
            )
        if verbose:
            print("model to qunatize:", self.model)
        if verbose:
            print("quantized model", self.mod_model)

        # self.print_size()
        return self.mod_model

    def reset(self):
        self.qconfig_dict = {

        }

    def set_config(self, config="fbgemm"):
        '''

        :param config: quantization configuration
        :return: no return, update the qconfig_dict
        '''
        if config == "fbgemm":
            self.mode = "fbgemm"
            self.qconfig = get_default_qconfig("fbgemm")
        else:
            self.mode = "none"
            self.qconfig = None

    @property
    def operatable(self):
        ret_list = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
                ret_list.append(name)
        return ret_list
