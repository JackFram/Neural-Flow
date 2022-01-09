import os
import torch
import numpy as np
import torch.nn as nn
from misc.train_bert import train_bert, time_model_evaluation
from opt import BertQuantizeOp, PruningOp, LowRankOp
import copy


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


def evaluate_solver(solver, get_solution_func, model_orig):
    acc = []
    f1 = []
    acc_f1 = []
    configs = solver.configs
    tokenizer = solver.tokenizer
    logger = solver.logger
    for storage_thresold in np.arange(solver.model_size, 0, -solver.model_size/10):

        model = copy.deepcopy(model_orig)

        print("Getting results for storage threshold {}".format(storage_thresold))

        solution = get_solution_func(storage_thresold)
        if solution is not None:
            quantize_list = []
            for layer in solution:
                for name in layer.split("+"):
                    layer_name, op_name, attrs = name.split("@")
                    if op_name == "lowrank":
                        op = LowRankOp(model)
                        model = op.apply([layer_name], rank=(int(attrs)))
                    elif op_name == "upruning":
                        op = PruningOp(model)
                        model = op.apply_([layer_name], amount=float(attrs))
                    elif op_name == "quantize" and attrs != "none":
                        quantize_list.append(layer_name)

            model.to("cuda")
            configs.device = "gpu"
            train_bert(configs, model, tokenizer, logger)
            model.eval()
            model.to("cpu")
            configs.device = "cpu"
            op = BertQuantizeOp(model)
            op.set_config()
            if len(quantize_list) > 0:
                mod_model = op.apply(name_list=quantize_list, verbose=False)
            else:
                mod_model = model
            results = time_model_evaluation(mod_model, configs, tokenizer, logger)
            acc.append(results["acc"])
            f1.append(results["f1"])
            acc_f1.append(results["acc_and_f1"])
    return acc, f1, acc_f1


def get_score(results, length=None, type="AUC"):
    results = np.array(results)
    if type == "AUC":
        if length == None:
            length = len(results)
        return results[:length].sum()*2 - results[0] - results[-1]
