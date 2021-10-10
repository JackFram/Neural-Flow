import copy
import torch
import torch.nn as nn
from .base import BaseOp

from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig


class QuantizeOp(BaseOp):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.op_name = "quantize"

    def apply(self, name=None, *kwargs):
        qconfig_dict = {
            "object_type": [
                (nn.Conv2d, float_qparams_weight_only_qconfig),
                (nn.Linear, default_dynamic_qconfig)
            ]
        }

        model_to_quantize = copy.deepcopy(self.model)
        prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
        print("prepared model:", prepared_model)
        self.mod_model = convert_fx(prepared_model)
        print("quantized model", self.mod_model)
        self.print_size()
