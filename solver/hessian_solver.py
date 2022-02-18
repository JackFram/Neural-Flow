from collections import defaultdict
import itertools
from .base_solver import *
from opt.utils import get_size
from opt import PruningOp, SPruningOp, BertQuantizeOp, LowRankOp, QuantizeOp
import torch.nn.utils.prune as prune
from misc.train_bert import get_bert_FIM
from misc.cv_utils import get_cv_FIM
from misc.translation import get_translation_FIM
import matplotlib.pyplot as plt
import cvxpy as cp
import pickle
import os
import random
import copy
import tqdm
from tqdm.contrib.itertools import product
from solver.utils import print_configuration, check_row


class OneShotHessianSolver(BaseSolver):
    def __init__(self, net, ops, configs=None, tokenizer=None, logger=None, task_name=None):
        super().__init__(net, ops)
        self.configs = configs
        self.tokenizer = tokenizer
        self.logger = logger
        self.task_name = task_name
        if "cifar10" in task_name or "image" in task_name:
            self.get_FIM_func = get_cv_FIM
        elif "MRPC" in task_name:
            self.get_FIM_func = get_bert_FIM
        elif "MarianMT-wmt16" in task_name:
            self.get_FIM_func = get_translation_FIM
        elif "t5-small-wmt16" in task_name:
            self.get_FIM_func = get_translation_FIM
        else:
            raise AttributeError("task name not existed")
        self._init()

    def _init(self):
        info_fn = f"./info/{self.task_name}-info.pkl"
        if os.path.exists(info_fn):
            with open(info_fn, "rb") as fp:
                info = pickle.load(fp)
                print(f"Load saved {self.task_name}-info of OSHS solver.")
                self.all_l = np.array(info["l"])
                self.all_name = info["name"]
                self.all_s = np.array(info["s"])
                self.model_size = self.all_s[:, 0].sum()
                self.cand = self.operatable
            return

        all_name,all_l,all_s = [], [], []
        self.cand = self.operatable
        for i, layer_name in enumerate(self.cand):
            if i != 3:
                continue
            print(f"Get profile count: {i+1}/{len(self.cand)} layer name: {layer_name}")
            FIM = self.get_FIM_func(self.configs, self.net, self.tokenizer, layer_name, self.logger)
            storage, original_size = self.get_storage_info(layer_name)

            op_indexed = defaultdict(list)
            for k, v in storage.items():
                _, op_name, attrs = k.split("@")
                op_indexed[op_name].append((k, v))  # name, storage

            layer_name, layer_l, layer_s = [], [], []
            for comb in tqdm.tqdm(product(op_indexed['lowrank'], op_indexed['spruning'], op_indexed['upruning'], op_indexed['quantize'])):
                name = "+".join([k for k, v in comb])
                layer_name.append(name)
                diff = self.get_diff(name)
                scale = self.get_scale_1(name_list=name)
                # score = (diff ** 2 * FIM).sum() * scale
                # print(FIM.mean(), FIM.max(), FIM.min())
                score = (diff ** 2 * FIM).sum()
                layer_l.append(score)
                layer_s.append(original_size * np.prod([max(0.00, v) for k, v in comb]))
            all_name.append(layer_name)
            all_l.append(layer_l)
            all_s.append(layer_s)
            print_configuration(all_name, all_l, all_s)
            exit(0)

        info = {"name": all_name, "s": all_s, "l": all_l}
        if "try" not in info_fn:
            with open(info_fn, "wb") as fp:
                pickle.dump(info, fp)
        self.all_l = np.array(all_l)
        self.all_name = all_name
        self.all_s = np.array(all_s)
        self.model_size = self.all_s[:, 0].sum()

    def get_param(self, mod:nn.modules):
        weight = mod.weight.data.cpu().numpy().flatten()
        if hasattr(mod, "bias") and mod.bias is not None:
            bias = mod.bias.data.cpu().numpy().flatten()
            return np.concatenate([weight, bias], axis=0)
        return np.concatenate([weight, np.zeros(mod.weight.data.cpu().numpy().shape[0])], axis=0)

    def get_storage_info(self, layer_name: str):
        storage = {}
        original_size = None
        layer_module = self.net.get_submodule(layer_name)
        original_size = self.get_param(layer_module).size
        for Op in self.ops:
            op = Op(self.net)
            if layer_name not in op.operatable:
                continue
            if isinstance(op, PruningOp) or isinstance(op, SPruningOp):
                for rate in np.arange(0, 6):
                    rate = 1 - 1/(2. ** rate)
                    name = f"{layer_name}@{op.op_name}@{rate:.2f}"
                    storage[name] = 1 - rate
            elif isinstance(op, BertQuantizeOp) or isinstance(op, QuantizeOp):
                for mode in [None, "fbgemm"]:
                    op.reset()
                    op.set_config(mode)
                    name = f"{layer_name}@{op.op_name}@{op.mode}"
                    if mode == "fbgemm":
                        storage[name] = get_size(torch.int8)
                    else:
                        storage[name] = get_size(torch.float32)
            elif isinstance(op, LowRankOp):
                for r in np.arange(0, 6):
                    rate = 1. / (2. ** r)
                    name = f"{layer_name}@{op.op_name}@{rate:.2f}"
                    storage[name] = rate
        return storage, original_size

    def get_layer_loss(self):
        layer_loss = self.all_l[:, -1] / self.all_s[:, 0]
        return layer_loss

    def get_assignment(self, storage_threshold: float, p_min:float=0.4, p_max:float=0.6):
        layer_loss = self.all_l[:, -1]
        mean_loss = layer_loss / self.all_s[:, 0]
        # print(f"mean loss: {mean_loss}")
        p = mean_loss / mean_loss.sum()
        p = p / (2*p.mean())
        print(f"p norm: {p}")
        # exit(0)
        p = np.clip(p, p_min, p_max)
        mem = (p*self.all_s[:, 0]).sum()

        p = (storage_threshold / mem) * p
        p[p >= 1.] = 1.
        assign_storage = p * self.all_s[:, 0]
        overflow_storage = storage_threshold - assign_storage.sum()

        while True:
            if np.isclose(overflow_storage, 0.):
                print(f"returned p: {p}, storage threshold: {assign_storage}")
                return assign_storage
            cand_flag = p < 1.
            mean_loss *= cand_flag

            p_inc = mean_loss / mean_loss.sum()
            p_inc = p_inc / (2 * p_inc.mean())
            p_inc = np.clip(p_inc, p_min, p_max)
            mem = (p_inc * self.all_s[:, 0]).sum()
            p_inc = (overflow_storage / mem) * p_inc
            p += p_inc
            p[p >= 1.] = 1.
            assign_storage = p * self.all_s[:, 0]
            overflow_storage = storage_threshold - assign_storage.sum()

    def get_gt(self, storage_threshold: float, p_min:float=0.4, p_max:float=0.6):
        l_layer_loss = np.load("layer.npz")["l_layer_loss"]
        mean_loss = l_layer_loss
        # print(f"mean loss: {mean_loss}")
        p = mean_loss / mean_loss.sum()
        p = p / (2*p.mean())
        print(f"p norm: {p}")
        # exit(0)
        p = np.clip(p, p_min, p_max)
        mem = (p*self.all_s[:, 0]).sum()

        p = (storage_threshold / mem) * p
        p[p >= 1.] = 1.
        assign_storage = p * self.all_s[:, 0]
        overflow_storage = storage_threshold - assign_storage.sum()

        while True:
            if np.isclose(overflow_storage, 0.):
                print(f"returned gt p: {p}, storage threshold: {assign_storage}")
                return assign_storage
            cand_flag = p < 1.
            mean_loss *= cand_flag

            p_inc = mean_loss / mean_loss.sum()
            p_inc = p_inc / (2 * p_inc.mean())
            p_inc = np.clip(p_inc, p_min, p_max)
            mem = (p_inc * self.all_s[:, 0]).sum()
            p_inc = (overflow_storage / mem) * p_inc
            p += p_inc
            p[p >= 1.] = 1.
            assign_storage = p * self.all_s[:, 0]
            overflow_storage = storage_threshold - assign_storage.sum()
    
    def should_skip(self, name:str, methods:set) -> bool:
        for name in name.split("+"):
            _, op_name, attrs = name.split("@")
            if op_name not in methods:
                if (op_name == "upruning" and float(attrs) != 0.00) or (
                    op_name == "quantize" and attrs != "none") or (
                    op_name == "lowrank" and float(attrs) != 1.00) or \
                   (op_name == "spruning" and float(attrs) != 0.00):
                    return True
        return False

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
        # print(p)
        solution = []
        for i in range(len(self.all_name)):
            upb = p[i]
            # upb = storage_threshold * p[i]
            best = None
            best_name = None
            # print(upb, self.all_l[i, :], self.all_s[i, :])
            for j in range(self.all_l.shape[1]):
                if self.all_s[i, j] <= upb+1e-4:
                    if best is None:
                        best = self.all_l[i, j]
                        best_name = self.all_name[i][j]
                    elif self.all_l[i, j] < best:
                        best = self.all_l[i, j]
                        best_name = self.all_name[i][j]
            solution.append(best_name)
        return solution

    def get_filtered_solution(self, storage_threshold: float, methods: set, use_gt=False):
        # p = np.ones((len(self.cand),)) * (min(storage_threshold, self.model_size)/self.model_size)
        if not use_gt:
            p = self.get_assignment(storage_threshold)
        else:
            p = self.get_gt(storage_threshold)
        # p = np.ones_like(p)
        # p[-1] = 0.05
        #
        # p[-1] = self.all_s[-1, 0]

        solution = []
        for i in range(len(self.all_name)):
            # upb = self.all_s[i, 0] * p[i]
            # print(upb, self.all_s[i, 0], p[i])
            upb = p[i]
            best = None
            best_name = None
            for j in range(self.all_l.shape[1]):
                # if i == len(self.all_name) - 1:
                #     print(upb, self.all_s[i, 0])
                if self.all_s[i, j] <= upb + 1e-4:
                    if i == len(self.all_name) - 1:
                        if self.should_skip(self.all_name[i][j], {"quantize", "lowrank"}):
                            continue
                    if self.should_skip(self.all_name[i][j], methods):
                        continue
                    # print(self.all_name[i][j])
                    # print(self.all_l[i, j])
                    # print(self.get_scale_1(self.all_name[i][j]))
                    if best is None or self.all_l[i, j] < best:
                        best = self.all_l[i, j]
                        best_name = self.all_name[i][j]
            # exit(0)
            if best_name is None:
                return None
            solution.append(best_name)
        return solution

    def get_layer_filtered_solution(self, storage_threshold: float, methods: set):
        # p = np.ones((len(self.cand),)) * (min(storage_threshold, self.model_size)/self.model_size)
        p = self.get_assignment(storage_threshold)
        solution_list = []

        p_rate = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        l_rate = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        q_rate = ["none", "fbgemm"]

        l_rate = [1.0]
        p_rate = [0.03125]
        q_rate = ["none"]

        for p in p_rate:
            for l in l_rate:
                for q in q_rate:
                    solution = []
                    for i in range(len(self.all_name)):
                        layer_name = self.all_name[i][0].split("+")[0].split("@")[0]
                        if i == 44:
                            operation = f"{layer_name}@lowrank@{l:.2f}+{layer_name}@spruning@0.00+{layer_name}@upruning@{1-p:.2f}+{layer_name}@quantize@{q}"
                            solution.append(operation)
                        else:
                            operation = f"{layer_name}@lowrank@1.00+{layer_name}@spruning@0.00+{layer_name}@upruning@0.00+{layer_name}@quantize@none"
                            solution.append(operation)
                    solution_list.append(solution)
        return solution_list

    def random_sample(self):
        # p = np.ones((len(self.cand),)) * (min(storage_threshold, self.model_size)/self.model_size)

        solution = []
        obj = 0.
        for i in range(len(self.all_name)):
            ind = random.randint(0, len(self.all_name[i])-1)
            name = self.all_name[i][ind]
            obj += self.all_l[i, ind]
            solution.append(name)
        return solution, obj

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

    def get_scale(self, name_list, eps=1e-2):
        scale = []
        for name in name_list.split("+"):
            layer_name, op_name, attrs = name.split("@")
            # print(f"processing {op_name} operator with attribute: {attrs}")
            if op_name == "upruning":
                scale.append(1./(eps + 1 - float(attrs)) ** 2)
            elif op_name == "quantize" and attrs != "none":
                if attrs == "none":
                    scale.append(1. / (eps + 1) ** 2)
                else:
                    scale.append(1. / (eps + 0.25) ** 2)
            elif op_name == "lowrank":
                scale.append(1. / (eps + float(attrs)) ** 2)
            elif op_name == "spruning":
                scale.append(1. / (eps + 1 - float(attrs)) ** 2)
        scale = np.max(np.array(scale))
        return scale

    def get_scale_1(self, name_list, eps=1e-2):
        scale = 1.
        for name in name_list.split("+"):
            layer_name, op_name, attrs = name.split("@")
            # print(f"processing {op_name} operator with attribute: {attrs}")
            if op_name == "upruning":
                scale *= 1 - float(attrs)
            elif op_name == "quantize" and attrs != "none":
                if attrs == "none":
                    scale *= 1
                else:
                    scale *= 0.25
            elif op_name == "lowrank":
                scale *= float(attrs)
            elif op_name == "spruning":
                scale *= 1 - float(attrs)
        scale = 1. / (eps + scale) ** 2
        return scale

    def get_diff(self, name_list):
        model = copy.deepcopy(self.net)
        # model.to("cpu")
        quantize_list = []

        for name in name_list.split("+"):
            layer_name, op_name, attrs = name.split("@")
            if op_name == "upruning":
                op = PruningOp(model)
                model = op.apply([layer_name], amount=float(attrs), remove_prune=True, inplace=True)
            elif op_name == "quantize" and attrs != "none":
                quantize_list.append(layer_name)
            elif op_name == "lowrank":
                op = LowRankOp(model)
                model = op.apply([layer_name], rank_fraction=float(attrs), inplace=True)
            elif op_name == "spruning":
                op = SPruningOp(model)
                model = op.apply([layer_name], amount=float(attrs), remove_prune=True, inplace=True)
        if len(quantize_list) > 0:
            # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            op = QuantizeOp(model)
            op.set_config()
            model = op.apply(name_list=quantize_list, verbose=False, inplace=True)

        if layer_name + ".SVDLinear-0" in op.operatable:
            mod_1 = model.get_submodule(layer_name + ".SVDLinear-0")
            mod_2 = model.get_submodule(layer_name + ".SVDLinear-1")
            if len(quantize_list) > 0:
                w_1 = mod_1.weight().dequantize().data.cpu().numpy()
                w_2 = mod_2.weight().dequantize().data.cpu().numpy()
            else:
                w_1 = mod_1.weight.data.cpu().numpy()
                w_2 = mod_2.weight.data.cpu().numpy()

            w = (w_2 @ w_1).flatten()
            if hasattr(mod_2, "bias") and mod_2.bias is not None and mod_2.bias() is not None:
                w = np.concatenate([w, mod_2.bias().dequantize().data.cpu().numpy().flatten()], axis=0)
            else:
                w = np.concatenate([w, np.zeros(w_2.shape[0])], axis=0)
        elif layer_name + ".SVDConv-0" in op.operatable:
            mod_1 = model.get_submodule(layer_name + ".SVDConv-0")
            mod_2 = model.get_submodule(layer_name + ".SVDConv-1")
            if len(quantize_list) > 0:
                w_1 = mod_1.weight().dequantize().data.cpu().numpy()
                w_2 = mod_2.weight().dequantize().data.cpu().numpy()
            else:
                w_1 = mod_1.weight.data.cpu().numpy()
                w_2 = mod_2.weight.data.cpu().numpy()

            w = (w_2 @ w_1).flatten()
            if hasattr(mod_2, "bias") and mod_2.bias is not None and mod_2.bias() is not None:
                w = np.concatenate([w, mod_2.bias().dequantize().data.cpu().numpy().flatten()], axis=0)
            else:
                w = np.concatenate([w, np.zeros(w_2.shape[0])], axis=0)
        else:
            mod = model.get_submodule(layer_name)
            if len(quantize_list) > 0:
                w = mod.weight().dequantize().data.cpu().numpy().flatten()
                if mod.bias() is not None:
                    w = np.concatenate([w, mod.bias().dequantize().data.cpu().numpy().flatten()], axis=0)
                else:
                    w = np.concatenate([w, np.zeros(mod.weight().dequantize().data.cpu().numpy().shape[0])], axis=0)
            else:
                w = self.get_param(mod)

        orig_w = self.get_param(self.net.get_submodule(layer_name))

        # if check_row(name_list)ß:
        #     print(name_list)
        #     print(orig_w[:10], w[:10])

        return orig_w - w