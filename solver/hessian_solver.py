from collections import defaultdict
import itertools
from .base_solver import *
from opt import PruningOp, SPruningOp, BertQuantizeOp, LowRankOp, QuantizeOp
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
        self._czy_init()

    def _czy_init(self):
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
            print(f"Get profile count: {i+1}/{len(self.cand)} layer name: {layer_name}")
            _, storage, diff, original_size, FIM = self.get_profile(layer_name)
            temp_dic = defaultdict(list)
            for k, v in diff.items():
                _, op_name, attrs = k.split("@")
                temp_dic[op_name].append((k, v, storage[k]))  # name, loss, storage

            layer_name, layer_l, layer_s = [], [], []
            for comb in tqdm.tqdm(itertools.product(temp_dic['lowrank'], temp_dic['spruning'], temp_dic['upruning'], temp_dic['quantize'])):
                name = "+".join([tup[0] for tup in comb])
                layer_name.append(name)
                diff = self.get_diff(name)
                layer_l.append((diff ** 2 * FIM).sum())
                layer_s.append(original_size * np.prod([max(0.00, tup[2]) for tup in comb]))
            all_name.append(layer_name)
            all_l.append(layer_l)
            all_s.append(layer_s)

        info = {"name": all_name, "s": all_s, "l": all_l}
        if "try" not in info_fn:
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
        original_size = None
        FIM = self.get_FIM_func(self.configs, self.net, self.tokenizer, layer_name, self.logger)
        for Op in self.ops:
            op = Op(self.net)
            if layer_name in op.operatable:
                if isinstance(op, PruningOp) or isinstance(op, SPruningOp):
                    # v = []
                    for rate in np.arange(0.00, 1.10, 0.10):
                        _, diff, storage_save = op.apply([layer_name], amount=rate, with_profile=True)
                        # obj = (diff[layer_name] ** 2 * FIM).sum()
                        # print(f"layer name: {layer_name}, {op.op_name} rate: {rate}, obj: {obj}.")
                        name = f"{layer_name}@{op.op_name}@{rate:.2f}"
                        # v.append(obj)
                        loss[name] = diff[layer_name]
                        storage[name] = storage_save[layer_name]
                        profile[name] = storage[name] / (loss[name] + 1e-12)
                        if original_size is None:
                            original_size = diff[layer_name].size
                    # plt.plot(np.arange(0.00, 1.05, 0.05), v, label=op.op_name)
                elif isinstance(op, BertQuantizeOp) or isinstance(op, QuantizeOp):
                    if isinstance(op, BertQuantizeOp):
                        op.model.to("cpu")
                    for mode in [None, "fbgemm"]:
                        op.reset()
                        op.set_config(mode)
                        _, diff, storage_save = op.apply([layer_name], with_profile=True)
                        # print(f"layer name: {layer_name}, quantization mode: {mode}, diff norm: {np.linalg.norm(diff[layer_name]*FIM)}.")
                        # obj = (diff[layer_name] ** 2 * FIM).sum()
                        name = f"{layer_name}@{op.op_name}@{op.mode}"
                        loss[name] = diff[layer_name]
                        storage[name] = storage_save[layer_name]
                        profile[name] = storage[name] / (loss[name] + 1e-12)
                        if original_size is None:
                            original_size = diff[layer_name].size
                elif isinstance(op, LowRankOp):
                    # v = []
                    for r in np.arange(1.00, 0.00, -0.10):
                        _, diff, storage_save = op.apply([layer_name], rank_fraction=r, with_profile=True)
                        # obj = (diff[layer_name] ** 2 * FIM).sum()
                        name = f"{layer_name}@{op.op_name}@{r:.2f}"
                        # v.append(obj)
                        loss[name] = diff[layer_name]
                        storage[name] = storage_save[layer_name]
                        profile[name] = storage[name] / (loss[name] + 1e-12)
                        if original_size is None:
                            original_size = diff[layer_name].size
        #             plt.plot(np.arange(0.00, 1.00, 0.05), v, label=op.op_name)
        # plt.legend()
        # plt.savefig("./results/test.pdf", bbox_inches="tight", dpi=500)
        # print(loss)
        # exit(0)
        return profile, storage, loss, original_size, FIM

    def get_assignment(self, storage_threshold: float, p_min:float=0.4, p_max:float=0.6):
        layer_loss = self.all_l[:, -1]
        mean_loss = layer_loss / self.all_s[:, 0]
        # print(f"mean loss: {mean_loss}")
        p = mean_loss / mean_loss.sum()
        p = p / (2*p.mean())
        p = np.clip(p, p_min, p_max)
        mem = (p*self.all_s[:, 0]).sum()

        p = (storage_threshold / mem) * p
        p[p >= 1.] = 1.
        assign_storage = p * self.all_s[:, 0]
        overflow_storage = storage_threshold - assign_storage.sum()

        while True:
            if np.isclose(overflow_storage, 0.):
                # print(f"returned p: {p}")
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

    def get_filtered_solution(self, storage_threshold: float, methods: set):
        # p = np.ones((len(self.cand),)) * (min(storage_threshold, self.model_size)/self.model_size)
        p = self.get_assignment(storage_threshold)
        solution = []
        for i in range(len(self.all_name)):
            # upb = self.all_s[i, 0] * p[i]
            upb = p[i]
            best = None
            best_name = None
            for j in range(self.all_l.shape[1]):
                if self.all_s[i, j] <= upb + 1e-4:
                    if self.should_skip(self.all_name[i][j], methods):
                        continue
                    # print(self.all_name[i][j])
                    # print(self.all_l[i, j])
                    if best is None or self.all_l[i, j] < best:
                        best = self.all_l[i, j]
                        best_name = self.all_name[i][j]
            if best_name is None:
                return None
            # print(best_name)
            # exit(0)
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

    def get_lowrank_solution(self, storage_threshold):
        p = np.ones((len(self.cand),)) * (min(storage_threshold, self.model_size)/self.model_size)
        # print(p)
        solution = []
        print("Getting solution for pure lowrank")
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

    def get_diff(self, name_list):
        model = copy.deepcopy(self.net)
        model.to("cpu")
        quantize_list = []
        for name in name_list.split("+"):
            layer_name, op_name, attrs = name.split("@")
            # print(f"processing {op_name} operator with attribute: {attrs}")
            if op_name == "upruning":
                op = PruningOp(model)
                model = op.apply([layer_name], amount=float(attrs), inplace=True)
                # w = model.get_submodule(layer_name + ".SVDLinear-0").weight.data.cpu().numpy().flatten()
                # print(op_name, (w == 0).sum() / w.shape[0])
            elif op_name == "quantize" and attrs != "none":
                quantize_list.append(layer_name)
            elif op_name == "lowrank":
                op = LowRankOp(model)
                model = op.apply([layer_name], rank_fraction=(float(attrs)), inplace=True)
            elif op_name == "spruning":
                op = SPruningOp(model)
                model = op.apply([layer_name], amount=float(attrs), inplace=True)
                # w = model.get_submodule(layer_name+".SVDLinear-0").weight.data.cpu().numpy().flatten()
        if len(quantize_list) > 0:
            # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            op = BertQuantizeOp(model)
            op.set_config()
            mod_model = op.apply(name_list=quantize_list, verbose=False, inplace=True)
        else:
            mod_model = model

        def get_param(mod: nn.modules):
            weight = mod.weight.data.cpu().numpy().flatten()
            if hasattr(mod, "bias") and mod.bias is not None:
                bias = mod.bias.data.cpu().numpy().flatten()
                return np.concatenate([weight, bias], axis=0)
            return weight

        orig_mod = self.net.get_submodule(layer_name)
        orig_w = get_param(orig_mod)

        if layer_name + ".SVDLinear-0" in op.operatable:
            mod_1 = mod_model.get_submodule(layer_name + ".SVDLinear-0")
            mod_2 = mod_model.get_submodule(layer_name + ".SVDLinear-1")
            if len(quantize_list) > 0:
                w_1 = mod_1.weight().dequantize().data.cpu().numpy()
                w_2 = mod_2.weight().dequantize().data.cpu().numpy()
            else:
                w_1 = mod_1.weight.data.cpu().numpy()
                w_2 = mod_2.weight.data.cpu().numpy()

            w = (w_2 @ w_1).flatten()
            if hasattr(mod_2, "bias") and mod_2.bias is not None and mod_2.bias() is not None:
                w = np.concatenate([w, mod_2.bias().dequantize().data.cpu().numpy().flatten()], axis=0)
        elif layer_name + ".SVDConv-0" in op.operatable:
            mod_1 = mod_model.get_submodule(layer_name + ".SVDConv-0")
            mod_2 = mod_model.get_submodule(layer_name + ".SVDConv-1")
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
            if len(quantize_list) > 0:
                mod = mod_model.get_submodule(layer_name)
                w = mod.weight().dequantize().data.cpu().numpy().flatten()
                if hasattr(mod, "bias") and mod.bias() is not None:
                    w = np.concatenate([w, mod.bias().dequantize().data.cpu().numpy().flatten()], axis=0)
            else:
                mod = mod_model.get_submodule(layer_name)
                w = get_param(mod)

        return orig_w - w