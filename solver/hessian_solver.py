from .base_solver import *
from opt import PruningOp, SPruningOp, BertQuantizeOp
from misc.train_bert import get_bert_FIM


class OneShotHessianSolver(BaseSolver):
    def __init__(self, net, ops, configs=None, tokenizer=None, logger=None):
        super().__init__(net, ops)
        self.configs = configs
        self.tokenizer = tokenizer
        self.logger = logger

    def get_profile(self, layer_name):
        profile = {}
        for Op in self.ops:
            op = Op(self.net)
            if layer_name in op.operatable:
                FIM = get_bert_FIM(self.configs, self.net, self.tokenizer, layer_name, self.logger)
                if isinstance(op, PruningOp) or isinstance(op, SPruningOp):
                    for rate in np.arange(0.00, 1.05, 0.05):
                        _, diff, storage_save = op.apply([layer_name], amount=rate, with_profile=True)
                        obj = ((diff[layer_name] * FIM) ** 2).sum()
                        score = storage_save[layer_name] / obj
                        name = f"{op.op_name}_{layer_name}_{rate:.2f}"
                        profile[name] = score
                elif isinstance(op, BertQuantizeOp):
                    op.set_config()
                    _, diff, storage_save = op.apply([layer_name], with_profile=True)
                    obj = ((diff[layer_name] * FIM) ** 2).sum()
                    score = storage_save[layer_name]/ obj
                    name = f"{op.op_name}_{layer_name}_{op.mode}"
                    profile[name] = score
        return profile

    def get_solution(self, storage, latency):
        return