import copy
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseOp
from collections import OrderedDict
from misc.train import train_model

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

    def apply(self, name_list: list=None, verbose=False, *args, **kwargs):

        '''

        :param name_list:
        :param verbose:
        :param args:
        :param kwargs:
        :return:
        '''
        if name_list is None:
            self.qconfig_dict = {"object_type": [(nn.Linear, self.qconfig)]}
        else:
            for name in name_list:

                if name not in self.operatable:
                    print("{} is not a quantizable layer, retry something in:{} !".format(name, self.operatable))
                    raise AttributeError

                self.qconfig_dict["module_name"][name] = self.qconfig
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
            "module_name": OrderedDict()
        }

    def apply_with_finetune(self, name_list, verbose=False, *args, **kwargs):
        for name in name_list:

            if name not in self.operatable:
                print("{} is not a quantizable layer, retry something in:{} !".format(name, self.operatable))
                raise AttributeError

            self.qconfig_dict["module_name"][name] = self.qconfig
        model_to_quantize = copy.deepcopy(self.model)
        if verbose:
            print("model to qunatize:", model_to_quantize)
        prepared_model = prepare_fx(model_to_quantize, self.qconfig_dict)

        print("Finetuning...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        prepared_model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(prepared_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.3)
        prepared_model = train_model(prepared_model, criterion, optimizer_ft, exp_lr_scheduler,
                                     num_epochs=10, device=device)

        if verbose:
            print("prepared model:", prepared_model)
        self.mod_model = convert_fx(prepared_model)
        if verbose:
            print("quantized model", self.mod_model)
        self.print_size()

        return self.mod_model

    def set_config(self, config=get_default_qconfig("fbgemm")):
        '''

        :param config: quantization configuration
        :return: no return, update the qconfig_dict
        '''
        self.qconfig = config

    @property
    def operatable(self):
        ret_list = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
                ret_list.append(name)
        return ret_list
