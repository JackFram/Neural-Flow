import numpy as np
import torch
import torch.nn as nn
from prettytable import PrettyTable


def print_configuration(all_name, all_l, all_s):
    for i, layer in enumerate(all_name):
        name_row = ["Layer", "lowrank", "sp", "up", "quantize", "loss", "storage"]
        t = PrettyTable(name_row)

        for j, config_name in enumerate(layer):
            if check_row(config_name):
                r = add_row(config_name)
                r.append(all_l[i][j])
                r.append(all_s[i][j])
                t.add_row(r)
        print(t)


def should_skip(name:str, methods:set) -> bool:
    for name in name.split("+"):
        _, op_name, attrs = name.split("@")
        if op_name not in methods:
            if (op_name == "upruning" and float(attrs) != 0.00) or (
                            op_name == "quantize" and attrs != "none") or (
                            op_name == "lowrank" and float(attrs) != 1.00) or \
                    (op_name == "spruning" and float(attrs) != 0.00):
                return True
    return False


def check_row(config_name):
    if not should_skip(config_name, {"lowrank"}):
        return True
    if not should_skip(config_name, {"upruning"}):
        return True
    if not should_skip(config_name, {"spruning"}):
        return True
    if not should_skip(config_name, {"quantize"}):
        return True

    if not should_skip(config_name, {"lowrank", "upruning"}):
        return True


def add_row(config_name):
    r = []
    layer_name = (config_name.split("+"))[0].split("@")[0]
    r.append(layer_name)
    for name in config_name.split("+"):
        layer_name, op_name, attrs = name.split("@")
        if op_name == "upruning":
            r.append(round(1. - float(attrs), 2))
        elif op_name == "quantize":
            if attrs != "none":
                r.append(0.25)
            else:
                r.append(1)
        elif op_name == "lowrank":
            r.append(float(attrs))
        elif op_name == "spruning":
            r.append(round(1. - float(attrs), 2))
    return r