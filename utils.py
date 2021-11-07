import os
import torch
import numpy as np
import torch.nn as nn


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def avg_deviation(arr1, arr2):
    # arr shape (out, in)
    dev = np.linalg.norm(arr1-arr2, axis=1)
    return dev.mean()


def model_deviation(model, mod_model):
    model_list = []
    mod_model_list = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # print(name, module.weight.detach().numpy().shape)
            model_list.append(module.weight.detach().numpy())

    for name, module in mod_model.named_modules():
        if isinstance(module, nn.quantized.dynamic.modules.linear.Linear):
            # print(name, module.weight().dequantize().detach().numpy().shape)
            mod_model_list.append(module.weight().dequantize().detach().numpy())
        elif isinstance(module, nn.Linear):
            mod_model_list.append(module.weight.detach().numpy())

    sum = 0

    for i in range(len(model_list)):
        t = avg_deviation(model_list[i], mod_model_list[i])
        sum += t

    return sum
