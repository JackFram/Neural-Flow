from collections import defaultdict
import itertools
from .base_solver import *
from opt import PruningOp, SPruningOp, BertQuantizeOp, LowRankOp
from misc.train_bert import get_bert_FIM
from misc.translation import get_translation_FIM
import matplotlib.pyplot as plt
import cvxpy as cp
import pickle
import os
import random


class OneShotHessianSolver(BaseSolver):
    def __init__(self, net, ops, configs=None, tokenizer=None, logger=None, task_name=None):
        super().__init__(net, ops)
        self.configs = configs
        self.tokenizer = tokenizer
        self.logger = logger
        self.task_name = task_name
        if task_name == "MRPC":
            self.get_FIM_func = get_bert_FIM
        elif task_name == "MarianMT-wmt16":
            self.get_FIM_func = get_translation_FIM
        elif task_name == "t5-small-wmt16":
            self.get_FIM_func = get_translation_FIM
        else:
            raise AttributeError("task name not existed")
        self._czy_init()

    def _init(self):
        info_fn = f"./{self.task_name}-info.pkl"
        if os.path.exists(info_fn):
            with open(info_fn, "rb") as fp:
                info = pickle.load(fp)
                print(f"Load saved {self.task_name}-info of OSHS solver.")
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
        for i, layer_name in enumerate(self.cand):
            print(f"Get profile count: {i+1}/{len(self.cand)}")
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
                _, op_name, attrs = k.split("@")
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
            print(name, l)
            all_name.append(name)
            all_l.append(l)
            all_s.append(s)
        info = {}
        info["name"] = all_name
        info["s"] = all_s
        info["l"] = all_l
        with open(info_fn, "wb") as fp:
            pickle.dump(info, fp)
        all_l = np.array(all_l)
        all_s = np.array(all_s)

        self.all_l = all_l
        self.all_name = all_name
        self.all_s = all_s
        self.model_size = self.all_s[:, 0].sum()

    def _czy_init(self):
        info_fn = f"./{self.task_name}-czy-info.pkl"
        if os.path.exists(info_fn):
            with open(info_fn, "rb") as fp:
                info = pickle.load(fp)
                print(f"Load saved {info_fn} of OSHS solver.")

                self.all_l = np.array(info["l"])
                self.all_name = info["name"]
                self.all_s = np.array(info["s"])
                self.model_size = self.all_s[:, 0].sum()
                self.cand = self.operatable
            return

        all_name,all_l,all_s = [], [], []
        self.cand = self.operatable
        for i, layer_name in enumerate(self.cand):
            print(f"Get profile count: {i+1}/{len(self.cand)}")
            _, storage, loss = self.get_profile(layer_name)
            temp_dic = defaultdict(list)
            for k, v in loss.items():
                _, op_name, attrs = k.split("@")
                temp_dic[op_name].append((k, v, storage[k]))  # name, loss, storage

            layer_name, layer_l, layer_s = [], [], []
            for comb in itertools.product(temp_dic['lowrank'], temp_dic['quantize']):
                layer_name.append("+".join([tup[0] for tup in comb]))
                layer_l.append(comb[0][1]+comb[1][1])
                layer_s.append(comb[0][2]*comb[1][2])
            all_name.append(layer_name)
            all_l.append(layer_l)
            all_s.append(layer_s)

        info = {"name": all_name, "s": all_s, "l": all_l}
        with open(info_fn, "wb") as fp:
            pickle.dump(info, fp)
        self.all_l = np.array(all_l)
        self.all_name = all_name
        self.all_s = np.array(all_s)
        self.model_size = self.all_s[:, 0].sum()

    def get_profile(self, layer_name: str):
        profile = {}
        storage = {}
        loss = {}
        for Op in self.ops:
            op = Op(self.net)
            if layer_name in op.operatable:
                FIM = self.get_FIM_func(self.configs, self.net, self.tokenizer, layer_name, self.logger)
                if isinstance(op, PruningOp) or isinstance(op, SPruningOp):
                    for rate in np.arange(0.00, 1.05, 0.05):
                        _, diff, storage_save = op.apply([layer_name], amount=rate, with_profile=True)
                        # print(f"layer name: {layer_name}, pruning rate: {rate}, diff norm: {np.linalg.norm(diff[layer_name]*FIM)}.")
                        obj = (diff[layer_name] ** 2 * FIM).sum()
                        name = f"{layer_name}@{op.op_name}@{rate:.2f}"
                        loss[name] = obj
                        storage[name] = storage_save[layer_name]
                        profile[name] = storage[name] / (loss[name] + 1e-12)
                elif isinstance(op, BertQuantizeOp):
                    op.model.to("cpu")
                    for mode in [None, "fbgemm"]:
                        op.reset()
                        op.set_config(mode)
                        _, diff, storage_save = op.apply([layer_name], with_profile=True)
                        # print(f"layer name: {layer_name}, quantization mode: {mode}, diff norm: {np.linalg.norm(diff[layer_name]*FIM)}.")
                        obj = (diff[layer_name] ** 2 * FIM).sum()
                        name = f"{layer_name}@{op.op_name}@{op.mode}"
                        loss[name] = obj
                        storage[name] = storage_save[layer_name]
                        profile[name] = storage[name] / (loss[name] + 1e-12)
                elif isinstance(op, LowRankOp):
                    for r in np.arange(0, 500, 50):
                        _, diff, storage_save = op.apply([layer_name], rank=r, with_profile=True)
                        obj = (diff[layer_name] ** 2 * FIM).sum()
                        name = f"{layer_name}@{op.op_name}@{r}"
                        loss[name] = obj
                        storage[name] = storage_save[layer_name]
                        profile[name] = storage[name] / (loss[name] + 1e-12)
        return profile, storage, loss

    def get_assignment(self, storage_threshold: float):
        mean_loss = self.all_l.mean(axis=1)
        assign = storage_threshold * mean_loss / mean_loss.sum()
        # mean_loss = np.log(mean_loss)
        # mean_loss = mean_loss - mean_loss.max()
        # exp_loss = np.exp(- temp * mean_loss)
        # # score = exp_loss / exp_loss.sum()
        while np.any(assign > self.all_s[:, 0]):
            # print(assign > self.all_s[:, 0])
            # print(assign)
            # print(assign.sum())
            zero_mask = assign <= self.all_s[:, 0]
            mean_loss *= zero_mask
            storage_overflow = (~zero_mask) * (assign - self.all_s[:, 0])
            storage_overflow = storage_overflow.sum()
            assign = np.minimum(assign, self.all_s[:, 0])
            if mean_loss.sum() == 0:
                break
            assign += storage_overflow * mean_loss / mean_loss.sum()

        return assign

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

        # p = np.ones((len(self.cand),)) * (min(storage_threshold, self.model_size)/self.model_size)
        p = self.get_assignment(storage_threshold)
        print(p)
        solution = []
        for i in range(len(self.all_name)):
            upb = p[i]
            # upb = storage_threshold * p[i]
            best = None
            best_name = None
            # print(upb, self.all_l[i, :], self.all_s[i, :])
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

        print("Getting MIP solution for pure quantization")

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

    def get_pruning_solution(self, storage_threshold):
        p = np.ones((len(self.cand),)) * (min(storage_threshold, self.model_size)/self.model_size)
        # print(p)
        solution = []
        print("Getting solution for pure pruning")
        for i in range(len(self.all_name)):
            upb = self.all_s[i, 0] * p[i]
            best = None
            best_name = None
            # print(upb, self.all_l[i, :], self.all_s[i, :])
            for j in range(self.all_l.shape[1]):
                if j % 2 == 0 and self.all_s[i, j] <= upb:
                    if best is None:
                        best = self.all_l[i, j]
                        best_name = self.all_name[i][j]
                    elif self.all_l[i, j] < best:
                        best = self.all_l[i, j]
                        best_name = self.all_name[i][j]
            solution.append(best_name)
        return solution

    def get_random_solution(self, storage_threshold):
        p = np.ones((len(self.cand),)) * (min(storage_threshold, self.model_size) / self.model_size)
        # print(p)
        solution = []
        print("Getting solution for random selection")
        for i in range(len(self.all_name)):
            upb = self.all_s[i, 0] * p[i]
            cand = []
            # print(upb, self.all_l[i, :], self.all_s[i, :])
            for j in range(self.all_l.shape[1]):
                if self.all_s[i, j] <= upb:
                    cand.append(self.all_name[i][j])
            solution.append(random.choice(cand))
        return solution

    def get_max_storage_solution(self, storage_threshold: float):

        p = np.ones((len(self.cand),)) * (min(storage_threshold, self.model_size)/self.model_size)
        # print(p)
        solution = []
        print("Getting solution for maximum storage selection")
        for i in range(len(self.all_name)):
            upb = self.all_s[i, 0] * p[i]
            best = None
            best_name = None
            # print(upb, self.all_l[i, :], self.all_s[i, :])
            for j in range(self.all_l.shape[1]):
                if self.all_s[i, j] <= upb:
                    if best is None:
                        best = self.all_s[i, j]
                        best_name = self.all_name[i][j]
                    elif self.all_s[i, j] > best:
                        best = self.all_s[i, j]
                        best_name = self.all_name[i][j]
            solution.append(best_name)
        return solution

        # layer_list = self.operatable
        #
        # prune_l = self.all_l[:, ::2]
        # prune_s = self.all_s[:, ::2]
        # print("Getting MIP solution for pure pruning")
        # print(prune_l[0, :], prune_s[0, :], self.all_name[0])
        #
        # select = cp.Variable(prune_l.shape, boolean=True)
        #
        # selection_constraint = cp.sum(select, axis=1) == 1.
        # storage_constraint = cp.sum(cp.multiply(prune_s, select)) <= storage_threshold
        # constraints = [storage_constraint, selection_constraint]
        # cost = cp.sum(cp.multiply(prune_l, select))
        #
        # problem = cp.Problem(cp.Minimize(cost), constraints=constraints)
        # m = problem.solve(verbose=True, cplex_params={"timelimit": 60})
        # solution = []
        # if select.value is not None:
        #     for i in range(len(self.all_name)):
        #         idx = np.where(select.value[i] == 1.)[0][0]
        #         solution.append(self.all_name[i][idx])
        #     return solution
        # else:
        #     return None