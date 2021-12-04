from .base_solver import *
from opt import PruningOp, SPruningOp, BertQuantizeOp
from misc.train_bert import get_bert_FIM
import matplotlib.pyplot as plt
import cvxpy as cp


class OneShotHessianSolver(BaseSolver):
    def __init__(self, net, ops, configs=None, tokenizer=None, logger=None):
        super().__init__(net, ops)
        self.configs = configs
        self.tokenizer = tokenizer
        self.logger = logger

    def get_profile(self, layer_name: str):
        profile = {}
        storage = {}
        loss = {}
        for Op in self.ops:
            op = Op(self.net)
            if layer_name in op.operatable:
                FIM = get_bert_FIM(self.configs, self.net, self.tokenizer, layer_name, self.logger)
                if isinstance(op, PruningOp) or isinstance(op, SPruningOp):
                    for rate in np.arange(0.00, 1.05, 0.05):
                        _, diff, storage_save = op.apply([layer_name], amount=rate, with_profile=True)
                        obj = ((diff[layer_name] * FIM) ** 2).sum()
                        name = f"{layer_name}_{op.op_name}_{rate:.2f}"
                        loss[name] = obj
                        storage[name] = storage_save[layer_name]
                        profile[name] = storage[name] / (loss[name] + 1e-12)
                elif isinstance(op, BertQuantizeOp):
                    op.set_config()
                    _, diff, storage_save = op.apply([layer_name], with_profile=True)
                    obj = ((diff[layer_name] * FIM) ** 2).sum()
                    name = f"{layer_name}_{op.op_name}_{op.mode}"
                    loss[name] = obj
                    storage[name] = storage_save[layer_name]
                    profile[name] = storage[name] / (loss[name] + 1e-12)
        return profile, storage, loss

    def get_solution(self, storage_threshold: float):
        layer_list = self.operatable
        all_name_p = []
        all_name_q = []
        all_l_p = []
        all_l_q = []
        all_s_p = []
        all_s_q = []
        for layer_name in layer_list[2:3]:
            profile, storage, loss = self.get_profile(layer_name)  # 0.35, 0.50, 0.30
            name_p = []
            name_q = []
            l_p = []
            l_q = []
            s_p = []
            s_q = []
            for k, v in loss.items():
                _, op_name, attrs = k.split("_")
                print(op_name)
                if op_name == "upruning":
                    name_p.append(k)
                    l_p.append(v)
                    s_p.append(storage[k])
                elif op_name == "quantize":
                    name_q.append(k)
                    l_q.append(v)
                    s_q.append(storage[k])
            all_name_p.append(name_p)
            all_name_q.append(name_q)
            all_l_p.append(l_p)
            all_l_q.append(l_q)
            all_s_p.append(s_p)
            all_s_q.append(s_q)
        all_l_q = np.array(all_l_q)
        all_l_p = np.array(all_l_p)
        all_s_q = np.array(all_s_q)
        all_s_p = np.array(all_s_p)

        P = cp.Variable(all_l_p.shape, boolean=True)
        Q = cp.Variable(all_l_q.shape, boolean=True)

        selection_constraint_P = cp.sum(P, axis=1) == 1
        selection_constraint_Q = cp.sum(Q, axis=1) == 1
        storage_constraint = cp.sum(cp.multiply(cp.sum(cp.multiply(all_s_q, Q), axis=1), cp.sum(cp.multiply(all_s_p, P), axis=1))) <= storage_threshold
        constraints = [selection_constraint_P, selection_constraint_Q, storage_constraint]
        cost = cp.sum(cp.multiply(all_l_p, P)) + cp.sum(cp.multiply(all_l_q, Q))

        problem = cp.Problem(cp.Minimize(cost), constraints=constraints)
        problem.solve(solver=cp.GLPK_MI)

        print(P.value, Q.value)


        print(all_l_p, all_l_q, all_s_p, all_s_q)

        return