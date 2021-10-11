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
        self.quantizable()

    def apply(self, verbose=False, *args, **kwargs):

        '''

        :param args:
        :param kwargs:
        :return:
        '''

        model_to_quantize = copy.deepcopy(self.model)
        prepared_model = prepare_fx(model_to_quantize, self.qconfig_dict)

        if verbose:
            print("prepared model:", prepared_model)
        self.mod_model = convert_fx(prepared_model)
        if verbose:
            print("quantized model", self.mod_model)
        self.print_size()

    def get_config(self, name_list: list, qconfig=get_default_qconfig("fbgemm")):
        '''

        :param name_list: module names to apply quantization
        :param qconfig: quantization configuration
        :return: no return, update the qconfig_dict
        '''
        self.qconfig_dict = {
            "module_name": [
            ]
        }

        for name in name_list:
            self.qconfig_dict["module_name"].append((name.replace("_", "."), qconfig))

    @property
    def quantizable(self):
        ret_list = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
                ret_list.append(name)
