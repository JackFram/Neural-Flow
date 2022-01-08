from .base_solver import *
from opt import PruningOp, SPruningOp, BertQuantizeOp
from misc.train_bert_emotion import get_bert_FIM
import matplotlib.pyplot as plt
import cvxpy as cp
import pickle
import os

info_path = "./info-emotion.pkl"

class OneShotHessianSolverEmotion(BaseSolver):
    def __init__(self, net, ops, configs=None, tokenizer=None, logger=None):
        super().__init__(net, ops)
        self.configs = configs
        self.tokenizer = tokenizer
        self.logger = logger
        self._init()

    def _init(self):
        if os.path.exists(info_path):
            with open(info_path, "rb") as fp:
                info = pickle.load(fp)
                print("Load saved info of OSHS solver.")
                all_l = np.array(info["l"])
                all_s = np.array(info["s"])

                self.all_l = all_l
                self.all_name = info["name"]
                self.all_s = all_s
                self.model_size = self.all_s[:, 0].sum()
                self.cand = self.operatable
            return
        layer_list = self.operatable
        all_name = []
        all_l = []
        all_s = []
        temp = 1
        self.cand = layer_list
        for layer_name in self.cand:
            profile, storage, loss = self.get_profile(layer_name)  # 0.35, 0.50, 0.30
            name_p = []
            name_q = []
            l_p = []
            l_q = []
            s_p = []
            s_q = []
            name = []
            l = []
            s = []
            for k, v in loss.items():
                _, op_name, attrs = k.split("_")
                if op_name == "upruning":
                    name_p.append(k)
                    l_p.append(v)
                    s_p.append(storage[k])
                elif op_name == "quantize":
                    name_q.append(k)
                    l_q.append(v)
                    s_q.append(storage[k])
            for i, n_p in enumerate(name_p):
                for j, n_q in enumerate(name_q):
                    name.append(n_p + "+" + n_q)
                    l.append(l_p[i]+l_q[j])
                    s.append(s_p[i]*s_q[j])
            all_name.append(name)
            all_l.append(l)
            all_s.append(s)
        info = {}
        info["name"] = all_name
        info["s"] = all_s
        info["l"] = all_l
        with open(info_path, "wb") as fp:
            pickle.dump(info, fp)
        all_l = np.array(all_l)
        all_s = np.array(all_s)

        self.all_l = all_l
        self.all_name = all_name
        self.all_s = all_s
        self.model_size = self.all_s[:, 0].sum()

    def get_profile(self, layer_name: str):
        profile = {}
        storage = {}
        loss = {}                
        FIM = get_bert_FIM(self.configs, self.net, self.tokenizer, layer_name, self.logger)
        for Op in self.ops:
            op = Op(self.net)
            if layer_name in op.operatable:
                if isinstance(op, PruningOp) or isinstance(op, SPruningOp):
                    for rate in np.arange(0.00, 1.05, 0.05):
                        _, diff, storage_save = op.apply([layer_name], amount=rate, with_profile=True)
                        # print(f"layer name: {layer_name}, pruning rate: {rate}, diff norm: {np.linalg.norm(diff[layer_name]*FIM)}.")
                        obj = (diff[layer_name] ** 2 * FIM).sum()
                        name = f"{layer_name}_{op.op_name}_{rate:.2f}"
                        loss[name] = obj
                        storage[name] = storage_save[layer_name]
                        profile[name] = storage[name] / (loss[name] + 1e-12)
                elif isinstance(op, BertQuantizeOp):
                    for mode in [None, "fbgemm"]:
                        op.reset()
                        op.set_config(mode)
                        _, diff, storage_save = op.apply([layer_name], with_profile=True)
                        # print(f"layer name: {layer_name}, quantization mode: {mode}, diff norm: {np.linalg.norm(diff[layer_name]*FIM)}.")
                        obj = (diff[layer_name] ** 2 * FIM).sum()
                        name = f"{layer_name}_{op.op_name}_{op.mode}"
                        loss[name] = obj
                        storage[name] = storage_save[layer_name]
                        profile[name] = storage[name] / (loss[name] + 1e-12)
        return profile, storage, loss

    def get_solution(self, storage_threshold: float):

        select = cp.Variable(self.all_l.shape, boolean=True)

        selection_constraint = cp.sum(select, axis=1) == 1.
        storage_constraint = cp.sum(cp.multiply(self.all_s, select)) <= storage_threshold
        constraints = [storage_constraint, selection_constraint]
        cost = cp.sum(cp.multiply(self.all_l, select))

        problem = cp.Problem(cp.Minimize(cost), constraints=constraints)
        m = problem.solve(verbose=True)

        return

    def get_zzh_solution(self, storage_threshold: float):

        # score = self.all_l[:, -1]
        #
        # assign = cp.Variable(len(self.cand))
        #
        # storage_constraint_1 = cp.sum(cp.multiply(assign, self.all_s[:, 0])) <= storage_threshold
        # storage_constraint_2 = assign <= 1.
        # storage_constraint_3 = assign >= 0.
        #
        # constraints = [storage_constraint_1, storage_constraint_2, storage_constraint_3]
        #
        # cost = score.T * cp.inv_pos(assign)
        #
        # problem = cp.Problem(cp.Minimize(cost), constraints=constraints)
        # m = problem.solve(verbose=True)
        #
        # p = np.array(assign.value)

        p = np.ones((len(self.cand),)) * (min(storage_threshold, self.model_size)/self.model_size)
        solution = []
        for i in range(len(self.all_name)):
            upb = self.all_s[i, 0] * p[i]
            best = None
            best_name = None
            for j in range(self.all_l.shape[1]):
                if self.all_s[i, j] <= upb:
                    if best is None:
                        best = self.all_l[i, j]
                        best_name = self.all_name[i][j]
                    elif self.all_l[i, j] < best:
                        best = self.all_l[i, j]
                        best_name = self.all_name[i][j]
            solution.append(best_name)
        return solution

    def get_quantize_solution(self, storage_threshold):
        layer_list = self.operatable

        quant_l = self.all_l[:, :2]
        quant_s = self.all_s[:, :2]

        select = cp.Variable(quant_l.shape, boolean=True)

        selection_constraint = cp.sum(select, axis=1) == 1.
        storage_constraint = cp.sum(cp.multiply(quant_s, select)) <= storage_threshold
        constraints = [storage_constraint, selection_constraint]
        cost = cp.sum(cp.multiply(quant_l, select))

        problem = cp.Problem(cp.Minimize(cost), constraints=constraints)
        m = problem.solve(verbose=True)
        solution = []
        if select.value is not None:
            for i in range(len(self.all_name)):
                if int(select.value[i, 0]) == 1.:
                    solution.append(self.all_name[i][0])
                else:
                    solution.append(self.all_name[i][1])
            return solution
        else:
            return None