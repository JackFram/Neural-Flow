from opt.lowrank import LowRankOp
from .base_solver import *
from opt.utils import get_size
from opt import PruningOp, SPruningOp, BertQuantizeOp
from misc.train_bert import get_bert_FIM
import matplotlib.pyplot as plt
import pickle
import os
from os.path import exists


class BaselineSolver(BaseSolver):
    def __init__(self, net, ops, task, configs=None, tokenizer=None, logger=None):
        super().__init__(net, ops)
        self.configs = configs
        self.tokenizer = tokenizer
        self.logger = logger
        self.task = task
        self.get_all_layer_profile_and_cache()

    def get_profile(self, layer_list: list):
        profile = {}
        storage = {}
        loss = {}
        for layer_name in layer_list:
            for Op in self.ops:
                op = Op(self.net)
                if layer_name in op.operatable:
                    FIM = get_bert_FIM(self.configs, self.net, self.tokenizer, layer_name, self.logger)
                    if isinstance(op, PruningOp) or isinstance(op, SPruningOp):
                        for rate in np.arange(0.00, 1.05, 0.05):
                            _, diff, storage_save = op.apply([layer_name], amount=rate, with_profile=True)
                            obj = (diff[layer_name] ** 2 * FIM).sum()
                            name = f"{op.op_name}_{layer_name}_{rate:.2f}"
                            loss[name] = obj
                            storage[name] = storage_save[layer_name]
                            profile[name] = storage[name] / (loss[name] + 1e-12)
                    elif isinstance(op, BertQuantizeOp):
                        op.model.to("cpu")
                        for mode in [None, "fbgemm"]:
                            op.reset()
                            op.set_config(mode)
                            _, diff, storage_save = op.apply([layer_name], with_profile=True)
                            obj = (diff[layer_name] ** 2 * FIM).sum()
                            name = f"{layer_name}_{op.op_name}_{op.mode}"
                            loss[name] = obj
                            storage[name] = storage_save[layer_name]
                            profile[name] = storage[name] / (loss[name] + 1e-12)
                    elif isinstance(op, LowRankOp):
                        for r in np.arange(50, 500, 50):
                            _, diff, storage_save = op.apply([layer_name], rank=r, with_profile=True)
                            obj = (diff[layer_name] ** 2 * FIM).sum()
                            name = f"{op.op_name}_{layer_name}_{r:.2f}"
                            loss[name] = obj
                            storage[name] = storage_save[layer_name]
                            profile[name] = storage[name] / (loss[name] + 1e-12)
        return profile, storage, loss

    def get_all_layer_profile_and_cache(self):
        data_dir = "results/profileData/"
        try:
            self.profile = pickle.load(open(data_dir + f"{self.task}-profile.p", "rb"))
            self.storage = pickle.load(open(data_dir + f"{self.task}-storage.p", "rb"))
            self.loss = pickle.load(open(data_dir + f"{self.task}-loss.p", "rb"))
        except:
            if not exists(data_dir):
                os.mkdir(data_dir)
            self.profile, self.storage, self.loss = self.get_profile(self.operatable)
            pickle.dump(self.profile, open(data_dir + f"{self.task}-profile.p", "wb"))
            pickle.dump(self.storage, open(data_dir + f"{self.task}-storage.p", "wb"))
            pickle.dump(self.loss, open(data_dir + f"{self.task}-loss.p", "wb"))
        
        op = self.ops[0](self.net)
        self.model_size = sum([self.storage[f"upruning_{layer_name}_0.00"] for layer_name in op.operatable])

    def get_solution(self, storage_threshold):
        layer_list = self.operatable

        op = "upruning"
        total_storage = {}
        total_loss = {}
        for k, v in self.loss.items():
            op_name, layer_name, attrs = k.split("_")
            if op_name != op:
                continue
            v = np.log(v)
            s = self.storage[k]
            total_storage[attrs] = total_storage.get(attrs, 0) + s
            total_loss[attrs] = total_loss.get(attrs, 0) + v

        best_loss = 0
        best_rate = 0
        for a, v in total_storage.items():
            print(a, v, total_loss[a])
            if v <= storage_threshold and total_loss[a] < best_loss:
                best_loss = total_loss[a]
                best_rate = a
        print(f"best rate is: {best_rate}")
        solution = []
        for layer in layer_list:
            name = "_".join([layer, op, str(best_rate)])
            solution.append(name)
        return solution
    
    