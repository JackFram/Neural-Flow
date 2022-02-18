import copy
import numpy as np
import torch
import torch.nn as nn
from .base import BaseOp
from .utils import get_size
from collections import OrderedDict

from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, get_default_qconfig


class QuantizeOp(BaseOp):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.op_name = "quantize"
        self.qconfig = None
        self.name_list = None
        self.qconfig_dict = {
            "module_name": OrderedDict()
        }

    def apply(self, name_list: list, verbose=False, inplace=False, *args, **kwargs):

        '''

        :param name_list:
        :param verbose:
        :param with_diff:
        :param args:
        :param kwargs:
        :return:
        '''
        if inplace:
            self.mod_model = self.model
        else:
            self.mod_model = copy.deepcopy(self.model)
        if self.qconfig is None:
            return self.mod_model
        for name in name_list:
            if name + ".SVDLinear-0" in self.operatable:
                self.qconfig_dict["module_name"][name + ".SVDLinear-0"] = self.qconfig
                self.qconfig_dict["module_name"][name + ".SVDLinear-1"] = self.qconfig
            elif name + ".SVDConv-0" in self.operatable:
                self.qconfig_dict["module_name"][name + ".SVDConv-0"] = self.qconfig
                self.qconfig_dict["module_name"][name + ".SVDConv-1"] = self.qconfig
            elif name not in self.operatable:
                print("{} is not a quantizable layer, retry something in:{} !".format(name, self.operatable))
                raise AttributeError
            else:
                self.qconfig_dict["module_name"][name] = self.qconfig
        if verbose:
            print("model to qunatize:", self.mod_model)
        prepared_model = prepare_fx(self.mod_model.eval(), self.qconfig_dict)
        if verbose:
            print("prepared model:", prepared_model)
        self.mod_model = convert_fx(prepared_model)
        if verbose:
            print("quantized model", self.mod_model)
        return self.mod_model

    def reset(self):
        self.mode = "none"
        self.qconfig_dict = {
            "module_name": OrderedDict()
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