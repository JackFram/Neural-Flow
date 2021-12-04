from .base_solver import *
from opt import PruningOp, SPruningOp, BertQuantizeOp
from misc.train_bert import get_bert_FIM
import matplotlib.pyplot as plt


class BaselineSolver(BaseSolver):
    def __init__(self, net, ops, configs=None, tokenizer=None, logger=None):
        super().__init__(net, ops)
        self.configs = configs
        self.tokenizer = tokenizer
        self.logger = logger

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
                            obj = ((diff[layer_name] * FIM) ** 2).sum()
                            name = f"{op.op_name}_{layer_name}_{rate:.2f}"
                            loss[name] = obj
                            storage[name] = storage_save[layer_name]
                            profile[name] = storage[name] / (loss[name] + 1e-12)
                    elif isinstance(op, BertQuantizeOp):
                        op.set_config()
                        _, diff, storage_save = op.apply([layer_name], with_profile=True)
                        obj = ((diff[layer_name] * FIM) ** 2).sum()
                        name = f"{op.op_name}_{layer_name}_{op.mode}"
                        loss[name] = obj
                        storage[name] = storage_save[layer_name]
                        profile[name] = storage[name] / (loss[name] + 1e-12)
        return profile, storage, loss

    def get_solution(self, storage_threshold=100):
        layer_list = self.operatable
        profile, storage, loss = self.get_profile(layer_list)

        op = "upruning"
        total_storage = {}
        total_loss = {}
        for k, v in loss.items():
            op_name, layer_name, attrs = k.split("_")
            if op_name != op:
                continue
            v = np.log(v)
            s = storage[k]
            total_storage[attrs] = total_storage.get(attrs, 0) + s
            total_loss[attrs] = total_loss.get(attrs, 0) + v

        best_loss = 0
        best_rate = 0
        for a, v in total_storage.items():
            if v > storage_threshold and total_loss[a] < best_loss:
                best_loss =  total_loss[a]
                best_rate = a
        print(best_rate)
        plt.plot(np.arange(0.00, 1.05, 0.05), total_loss.values(), label="loss")
        plt.plot(np.arange(0.00, 1.05, 0.05), total_storage.values(), label="storage")
        plt.legend()
        plt.savefig(f"./results/random_{op}.pdf")
        return
    
    