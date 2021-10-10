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

    def apply(self, name=None, *args, **kwargs):

        '''

        :param name:
        :param args:
        :param kwargs:
        :return:
        '''

        model_to_quantize = copy.deepcopy(self.model)
        prepared_model = prepare_fx(model_to_quantize, self.qconfig_dict)
        print("prepared model:", prepared_model)
        self.mod_model = convert_fx(prepared_model)
        print("quantized model", self.mod_model)
        self.print_size()

    def get_config(self):
        '''

        qconfig = {
            " : qconfig_global,
            "sub" : qconfig_sub,
            "sub.fc" : qconfig_fc,
            "sub.conv": None
        }
        qconfig_dict = {
            # qconfig? means either a valid qconfig or None
            # optional, global config
            "": qconfig?,
            # optional, used for module and function types
            # could also be split into module_types and function_types if we prefer
            "object_type": [
                (torch.nn.Conv2d, qconfig?),
                (torch.nn.functional.add, qconfig?),
                ...,
            ],
            # optional, used for module names
            "module_name": [
                ("foo.bar", qconfig?)
                ...,
            ],
            # optional, matched in order, first match takes precedence
            "module_name_regex": [
                ("foo.*bar.*conv[0-9]+", qconfig?)
                ...,
            ],
            # priority (in increasing order): global, object_type, module_name_regex, module_name
            # qconfig == None means fusion and quantization should be skipped for anything
            # matching the rule

            # **api subject to change**
            # optional: specify the path for standalone modules
            # These modules are symbolically traced and quantized as one unit
            # so that the call to the submodule appears as one call_module
            # node in the forward graph of the GraphModule
            "standalone_module_name": [
                "submodule.standalone"
            ],
            "standalone_module_class": [
                StandaloneModuleClass
            ]
        }

        '''
        qconfig = get_default_qconfig("fbgemm")
        self.qconfig_dict = {
            "module_name": [
                ("conv1", qconfig),
                ("layer1.0.conv1", qconfig)
            ]
        }
