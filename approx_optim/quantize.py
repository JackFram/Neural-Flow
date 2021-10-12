import copy
import torch
import torch.nn as nn
from .base import BaseOp

from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, get_default_qconfig


class QuantizeOp(BaseOp):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.op_name = "quantize"
        self.qconfig_dict = None
        self.qconfig = None
        self.name_list = None
        self.qconfig_dict = {
            "module_name": [
            ]
        }

    def apply(self, name_list, verbose=False, *args, **kwargs):

        '''

        :param name_list:
        :param verbose:
        :param args:
        :param kwargs:
        :return:
        '''

        for name in name_list:
            if name not in self.name_list:
                print("{} is not a layer name, retry something in:{} !".format(name, self.operatable))
                raise AttributeError

            if name not in self.operatable:
                print("{} is not a quantizable layer, retry something in:{} !".format(name, self.operatable))
                raise AttributeError

            self.qconfig_dict["module_name"].append((name, self.qconfig))

        model_to_quantize = copy.deepcopy(self.model)
        if verbose:
            print("model to qunatize:", model_to_quantize)
        prepared_model = prepare_fx(model_to_quantize, self.qconfig_dict)

        if verbose:
            print("prepared model:", prepared_model)
        self.mod_model = convert_fx(prepared_model)
        if verbose:
            print("quantized model", self.mod_model)
        self.print_size()

        return self.mod_model

    def reset(self):
        self.qconfig_dict = {
            "module_name": [
            ]
        }

    def set_config(self, name_list: list, qconfig=get_default_qconfig("fbgemm")):
        '''

        :param name_list: module names to apply quantization
        :param qconfig: quantization configuration
        :return: no return, update the qconfig_dict
        '''
        self.name_list = name_list
        self.qconfig = qconfig

    @property
    def operatable(self):
        ret_list = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
                ret_list.append(name)
        return ret_list
