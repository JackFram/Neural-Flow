from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import random
import torch
import torch.nn as nn

from argparse import Namespace
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, get_default_qconfig
from misc.train_bert import train_bert, time_model_evaluation, get_bert_FIM
from utils import print_size_of_model, model_deviation, evaluate_solver
import seaborn as sns
import matplotlib.pylab as plt
import pickle


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

logging.getLogger("transformers.modeling_utils").setLevel(
   logging.WARN)  # Reduce logging

print(torch.__version__)

torch.set_num_threads(1)
print(torch.__config__.parallel_info())

configs = Namespace()

# The output directory for the fine-tuned model, $OUT_DIR.
configs.output_dir = "../data/MRPC_model/"

# The data directory for the MRPC task in the GLUE benchmark, $GLUE_DIR/$TASK_NAME.
configs.data_dir = "../data/glue_data/MRPC"

# The model name or path for the pre-trained model.
configs.model_name_or_path = "bert-base-uncased"
# The maximum length of an input sequence
configs.max_seq_length = 128

# Prepare GLUE task.
configs.task_name = "MRPC".lower()
configs.processor = processors[configs.task_name]()
configs.output_mode = output_modes[configs.task_name]
configs.label_list = configs.processor.get_labels()
num_labels = len(configs.label_list)
configs.model_type = "bert".lower()
configs.do_lower_case = True

# Set the device, batch size, topology, and caching flags.
configs.device = "gpu"
configs.per_device_eval_batch_size = 8
configs.per_device_train_batch_size = 8
configs.n_gpu = 0
configs.local_rank = -1
configs.overwrite_cache = False

# Set Training arguments
configs.learning_rate = 1e-7
configs.weight_decay = 0.0
configs.num_train_epochs = 3
configs.max_train_steps = None
configs.gradient_accumulation_steps = 1
configs.lr_scheduler_type = "linear"
configs.num_warmup_steps = 0

# Set random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

tokenizer = BertTokenizer.from_pretrained(
    configs.output_dir, do_lower_case=configs.do_lower_case)

model_orig = BertForSequenceClassification.from_pretrained(configs.output_dir)


# ####### Solver ##########
from solver import OneShotHessianSolver, BaselineSolver
from opt import BertQuantizeOp, SPruningOp, PruningOp, LowRankOp

# quant_acc = []
# oshs_acc = []
# prune_acc = []
# f1 = []
# acc_f1 = []

Ops = [LowRankOp]
# solver = OneShotHessianSolver(model_orig.eval(), Ops, configs, tokenizer, logger)
base_solver = BaselineSolver(model_orig.eval(), Ops, "testlowrank", configs, tokenizer, logger)
#hession_solver = OneShotHessianSolver(model_orig.eval(), Ops, configs, tokenizer, logger, task_name="MRPC")

# oshs_acc, _, _ = evaluate_solver(hession_solver, hession_solver.get_zzh_solution, model_orig)
# quant_acc, _, _ = evaluate_solver(hession_solver, hession_solver.get_quantize_solution, model_orig)
# prune_acc, _, _ = evaluate_solver(base_solver, base_solver.get_solution, model_orig)

# quant_range = np.arange(hession_solver.model_size, 0, -hession_solver.model_size/10)[:len(quant_acc)]
# oshs_range = np.arange(hession_solver.model_size, 0, -hession_solver.model_size/10)

# plt.plot(quant_range, quant_acc, label="pure_quant_acc")
# plt.plot(oshs_range, oshs_acc, label="oshs_acc")
# plt.plot(oshs_range, prune_acc, label="prune_acc")
# plt.legend()
# plt.savefig("./results/QP_OSHS_ft.pdf", bbox_inches="tight", dpi=500)


# print("Before optimization:")
# time_model_evaluation(model_orig, configs, tokenizer, logger)
#
# for storage_thresold in np.arange(solver.model_size, 0, -solver.model_size/10):
#
#     model = copy.deepcopy(model_orig)
#
#     print("Getting results for storage threshold {}".format(storage_thresold))
#
#     solution = solver.get_solution(storage_thresold)
#     if solution is not None:
#         quantize_list = []
#         for layer in solution:
#             for name in layer.split("+"):
#                 layer_name, op_name, attrs = name.split("_")
#                 if op_name == "upruning":
#                     op = PruningOp(model)
#                     model = op.apply_([layer_name], amount=float(attrs))
#                 elif op_name == "quantize" and attrs != "none":
#                     quantize_list.append(layer_name)
#
#         model.to("cuda")
#         configs.device = "gpu"
#         train_bert(configs, model, tokenizer, logger)
#         model.eval()
#         # model.to("cpu")
#         # configs.device = "cpu"
#         # op = BertQuantizeOp(model)
#         # op.set_config()
#         # print(quantize_list)
#         if len(quantize_list) > 0:
#             mod_model = op.apply(name_list=quantize_list, verbose=False)
#         else:
#             mod_model = model
#         results = time_model_evaluation(mod_model, configs, tokenizer, logger)
#         prune_acc.append(results["acc"])
#
#         # f1.append(results["f1"])
#         # acc_f1.append(results["acc_and_f1"])

# pickle.dump(prune_acc, open("./results/MRPC/" + "prune_acc.p", "wb"))

# for storage_thresold in np.arange(solver.model_size, 0, -solver.model_size/10):
#
#     model = copy.deepcopy(model_orig)
#
#     print("Getting results for storage threshold {}".format(storage_thresold))
#
#     solution = solver.get_zzh_solution(storage_thresold)
#     if solution is not None:
#         quantize_list = []
#         for layer in solution:
#             for name in layer.split("+"):
#                 layer_name, op_name, attrs = name.split("_")
#                 if op_name == "upruning":
#                     op = PruningOp(model)
#                     model = op.apply_([layer_name], amount=float(attrs))
#                 elif op_name == "quantize" and attrs != "none":
#                     quantize_list.append(layer_name)
#
#         model.to("cuda")
#         configs.device = "gpu"
#         train_bert(configs, model, tokenizer, logger)
#         model.eval()
#         model.to("cpu")
#         configs.device = "cpu"
#         op = BertQuantizeOp(model)
#         op.set_config()
#         if len(quantize_list) > 0:
#             mod_model = op.apply(name_list=quantize_list, verbose=False)
#         else:
#             mod_model = model
#         results = time_model_evaluation(mod_model, configs, tokenizer, logger)
#         oshs_acc.append(results["acc"])

# quant_acc = pickle.load(open("./results/MRPC/" + "quant_acc.p", "rb"))
# prune_acc = pickle.load(open("./results/MRPC/" + "prune_acc.p", "rb"))
# oshs_acc = pickle.load(open("./results/MRPC/" + "oshs_acc.p", "rb"))
# quant_range = np.arange(solver.model_size, 0, -solver.model_size/10)[:len(quant_acc)]
# oshs_range = np.arange(solver.model_size, 0, -solver.model_size/10)
# #
# plt.plot(quant_range, quant_acc, label="pure_quant_acc")
# plt.plot(oshs_range, oshs_acc, label="oshs_acc")
# plt.plot(oshs_range, prune_acc, label="prune_acc")
# plt.legend()
# plt.savefig("./results/QP_OSHS_ft.pdf", bbox_inches="tight", dpi=500)

# pickle.dump(quant_acc, open("./results/MRPC/" + "quant_acc.p", "wb"))
# pickle.dump(oshs_acc, open("./results/MRPC/" + "oshs_acc.p", "wb"))
# plt.plot(np.arange(solver.model_size, 0, -solver.model_size/10), acc, label="acc")
# plt.plot(np.arange(solver.model_size, 0, -solver.model_size/10), f1, label="f1")
# plt.plot(np.arange(solver.model_size, 0, -solver.model_size/10), acc_f1, label="acc+f1")
# plt.scatter(solver.model_size-255.2, 0.860, label="pure quantization")
# plt.legend()
# plt.savefig("./results/OSHS_ft.pdf", bbox_inches="tight", dpi=500)


# #########################


# ####### Quantization ##########
# from opt import QuantizeOp
# model_orig.to("cpu")
# op = QuantizeOp(model_orig)
# op.set_config()
# mod_model = op.apply(name_list=op.operatable, verbose=False)
# configs.device = "gpu"
# time_model_evaluation(mod_model.cuda(), configs, tokenizer, logger)
# ###############################


# ####### Pruning ##########

# from opt import PruningOp, SPruningOp
#
# op = PruningOp(model, amount=0.4)
# mod_model, diff, storage_save = op.apply(name_list=op.operatable[:3], verbose=False, with_profile=True)

# ##########################


# print("num labels: {}".format(num_labels))
#
# acc = []
# f1 = []
# acc_f1 = []
# dev = []
#
# for rate in np.arange(0.90, 1.05, 0.05):
#     print("rate:%f "%rate)
#     op = SPruningOp(model, amount=rate)
#     mod_model, diff = op.apply(name_list=op.operatable[:1], verbose=False, with_diff=True)
#     FIM = get_bert_FIM(configs, model, tokenizer, op.operatable[0], logger)
#     param = diff[op.operatable[0]]
#     train_bert(configs, mod_model, tokenizer, logger)
#     results = time_model_evaluation(mod_model, configs, tokenizer, logger)
#     acc.append(results["acc"])
#     f1.append(results["f1"])
#     acc_f1.append(results["acc_and_f1"])
#     dev.append(model_deviation(model, mod_model.to("cpu")))



#
# # plt.plot(np.arange(0, 1.05, 0.05), acc, label="acc")
# # plt.plot(np.arange(0, 1.05, 0.05), f1, label="f1")
# # plt.plot(np.arange(0, 1.05, 0.05), acc_f1, label="acc+f1")
# # plt.plot(np.arange(0, 1.05, 0.05), dev, label="deviation")
# # plt.savefig("./results/dev.pdf", bbox_inches="tight", dpi=500)
#
# # Create some mock data
# t = np.arange(0, 1.05, 0.05)
#
# fig, ax1 = plt.subplots()
#
# color = 'tab:red'
# ax1.set_xlabel('pruning rate')
# ax1.set_ylabel('accuracy and f1')
# ax1.plot(t, acc, label="acc")
# ax1.plot(t, f1, label="f1")
# ax1.plot(t, acc_f1, label="acc+f1")
# ax1.tick_params(axis='y')
# ax1.legend(loc='center left')
#
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:blue'
# ax2.set_ylabel('deviation')  # we already handled the x-label with ax1
# ax2.plot(t, dev, label="deviation")
# ax2.tick_params(axis='y')
# ax2.legend(loc='center right')
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig("./results/ft_sprune_l1_dev.pdf", bbox_inches="tight", dpi=500)
#
#
# print_size_of_model(model)
# print_size_of_model(mod_model)
#
# time_model_evaluation(model, configs, tokenizer, logger)
# time_model_evaluation(mod_model, configs, tokenizer, logger)
#
# train_bert(configs, mod_model, tokenizer, logger)
# time_model_evaluation(mod_model, configs, tokenizer, logger)
