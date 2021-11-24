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
from utils import print_size_of_model, model_deviation
import seaborn as sns
import matplotlib.pylab as plt


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
configs.device = "cpu"
configs.per_device_eval_batch_size = 8
configs.per_device_train_batch_size = 8
configs.n_gpu = 0
configs.local_rank = -1
configs.overwrite_cache = False

# Set Training arguments
configs.learning_rate = 2e-5
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

model = BertForSequenceClassification.from_pretrained(configs.output_dir)

# Quantization
from opt import BertQuantizeOp
op = BertQuantizeOp(model)
op.set_config()
mod_model, diff = op.apply(name_list=op.operatable[:1], verbose=False, with_diff=True)
param = diff[op.operatable[0]]
FIM = get_bert_FIM(configs, model, tokenizer, op.operatable[0], logger)
print(param[:20], FIM[:20])

# qconfig = get_default_qconfig("fbgemm")
#
# mod_model = torch.quantization.quantize_dynamic(
#     model, {nn.Linear}, dtype=torch.qint8  # qint8, float16, quint8
# )
#
# for name, module in mod_model.named_modules():
#     if isinstance(module, nn.quantized.dynamic.modules.linear.Linear):
#         print(name, module.bias().dequantize().data.cpu().numpy().flatten().shape)

# for name, module in model.named_modules():
#     if isinstance(module, nn.Linear):
#         print(name)


# ####### Pruning ##########

# from opt import PruningOp, SPruningOp
#
# print("num labels: {}".format(num_labels))
#
# acc = []
# f1 = []
# acc_f1 = []
# dev = []
#
# for rate in np.arange(1, 1.05, 0.05):
#     print("rate:%f "%rate)
#     op = PruningOp(model, amount=rate)
#     mod_model, diff = op.apply(name_list=op.operatable[:1], verbose=False, with_diff=True)
#     FIM = get_bert_FIM(configs, model, tokenizer, op.operatable[0], logger)
#     param = diff[op.operatable[0]]
#     print(param.shape, FIM.shape)
#     exit(0)
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
# ####### End of Pruning ##########
