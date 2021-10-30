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
from misc.train_bert import train_bert, time_model_evaluation
from utils import print_size_of_model


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

# args = parse_args()
#
# config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
#
#
# tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
# model = AutoModelForSequenceClassification.from_pretrained(
#     args.model_name_or_path,
#     from_tf=bool(".ckpt" in args.model_name_or_path),
#     config=config,
# )
# model.to('cpu')

# train_bert(args, model, tokenizer)

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Quantization
# qconfig = get_default_qconfig("fbgemm")

# for name, module in model.named_modules():
#     print(name)

# quantized_model = torch.quantization.quantize_dynamic(
#     model, {nn.Linear: qconfig}  # , dtype=torch.qint8
# )

# print(quantized_model)

# Pruning

from opt import PruningOp

op = PruningOp(model)
mod_model = op.apply(name_list=op.operatable)


print_size_of_model(model)
print_size_of_model(mod_model)

# Evaluate the original FP32 BERT model
time_model_evaluation(model, configs, tokenizer, logger)
time_model_evaluation(mod_model, configs, tokenizer, logger)

# Evaluate the INT8 BERT model after the dynamic quantization
train_bert(configs, mod_model, tokenizer, logger)
time_model_evaluation(mod_model, configs, tokenizer, logger)

