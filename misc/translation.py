import argparse
import logging
import math
import os
import random
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MBartTokenizer,
    MBartTokenizerFast,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
from argparse import Namespace


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# Parsing input arguments
def parse_args():

    # parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    args = Namespace()
    args.dataset_name = "wmt16"
    args.predict_with_generate = True
    args.dataset_config_name = "ro-en"
    args.train_file = None
    args.num_beams = None
    args.max_source_length = 512
    args.max_target_length = 128
    args.val_max_target_length = None
    args.pad_to_max_length = False

    # parser.add_argument(
    #     "--dataset_name",
    #     type=str,
    #     default=None,
    #     help="The name of the dataset to use (via the datasets library).",
    # )
    #
    # parser.add_argument(
    #     "--predict_with_generate",
    #     type=bool,
    #     default=True,
    #     help="",
    # )
    # parser.add_argument(
    #     "--dataset_config_name",
    #     type=str,
    #     default=None,
    #     help="The configuration name of the dataset to use (via the datasets library).",
    # )
    # parser.add_argument(
    #     "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    # )
    #
    # parser.add_argument(
    #     "--num_beams",
    #     type=int,
    #     default=None,
    #     help="Number of beams to use for evaluation. This argument will be "
    #     "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    # )
    #
    # parser.add_argument(
    #     "--max_source_length",
    #     type=int,
    #     default=1024,
    #     help="The maximum total input sequence length after "
    #     "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    # )
    # parser.add_argument(
    #     "--max_target_length",
    #     type=int,
    #     default=128,
    #     help="The maximum total sequence length for target text after "
    #     "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
    #     "during ``evaluate`` and ``predict``.",
    # )
    # parser.add_argument(
    #     "--val_max_target_length",
    #     type=int,
    #     default=None,
    #     help="The maximum total sequence length for validation "
    #     "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
    #     "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
    #     "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    # )
    # parser.add_argument(
    #     "--pad_to_max_length",
    #     type=bool,
    #     default=False,
    #     help="Whether to pad all samples to model maximum sentence "
    #     "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
    #     "efficient on GPU but very bad for TPU.",
    # )
    args.validation_file = None
    # parser.add_argument(
    #     "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    # )
    args.ignore_pad_token_for_loss = True
    # parser.add_argument(
    #     "--ignore_pad_token_for_loss",
    #     type=bool,
    #     default=True,
    #     help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    # )
    args.source_lang = "en"
    # parser.add_argument("--source_lang", type=str, default=None, help="Source language id for translation.")
    args.target_lang = "ro"
    # parser.add_argument("--target_lang", type=str, default=None, help="Target language id for translation.")
    args.source_prefix = None
    # parser.add_argument(
    #     "--source_prefix",
    #     type=str,
    #     default=None,
    #     help="A prefix to add before every source text " "(useful for T5 models).",
    # )
    args.preprocessing_num_workers = None
    # parser.add_argument(
    #     "--preprocessing_num_workers",
    #     type=int,
    #     default=None,
    #     help="The number of processes to use for the preprocessing.",
    # )
    args.overwrite_cache = None
    # parser.add_argument(
    #     "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    # )
    args.max_length = 128
    # parser.add_argument(
    #     "--max_length",
    #     type=int,
    #     default=128,
    #     help=(
    #         "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
    #         " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    #     ),
    # )
    args.model_name_or_path = "Helsinki-NLP/opus-mt-en-ro"
    # parser.add_argument(
    #     "--model_name_or_path",
    #     type=str,
    #     help="Path to pretrained model or model identifier from huggingface.co/models.",
    #     required=True,
    # )
    args.config_name = None
    # parser.add_argument(
    #     "--config_name",
    #     type=str,
    #     default=None,
    #     help="Pretrained config name or path if not the same as model_name",
    # )
    args.tokenizer_name = None
    # parser.add_argument(
    #     "--tokenizer_name",
    #     type=str,
    #     default=None,
    #     help="Pretrained tokenizer name or path if not the same as model_name",
    # )
    args.use_slow_tokenizer = None
    # parser.add_argument(
    #     "--use_slow_tokenizer",
    #     action="store_true",
    #     help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    # )
    args.per_device_train_batch_size = 8
    # parser.add_argument(
    #     "--per_device_train_batch_size",
    #     type=int,
    #     default=8,
    #     help="Batch size (per device) for the training dataloader.",
    # )
    args.per_device_eval_batch_size = 8
    # parser.add_argument(
    #     "--per_device_eval_batch_size",
    #     type=int,
    #     default=8,
    #     help="Batch size (per device) for the evaluation dataloader.",
    # )
    args.learning_rate = 5e-5
    # parser.add_argument(
    #     "--learning_rate",
    #     type=float,
    #     default=5e-5,
    #     help="Initial learning rate (after the potential warmup period) to use.",
    # )
    args.weight_decay = 0.0
    # parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    args.num_train_epochs = 3
    # parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    args.max_train_steps = None
    # parser.add_argument(
    #     "--max_train_steps",
    #     type=int,
    #     default=None,
    #     help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    # )
    args.gradient_accumulation_steps = 1
    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     type=int,
    #     default=1,
    #     help="Number of updates steps to accumulate before performing a backward/update pass.",
    # )
    args.lr_scheduler_type = "linear"
    # parser.add_argument(
    #     "--lr_scheduler_type",
    #     type=SchedulerType,
    #     default="linear",
    #     help="The scheduler type to use.",
    #     choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    # )
    args.num_warmup_steps = 0
    # parser.add_argument(
    #     "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    # )
    args.output_dir = "tst-translation"
    # parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    args.seed = None
    # parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args.model_type = None
    # parser.add_argument(
    #     "--model_type",
    #     type=str,
    #     default=None,
    #     help="Model type to use if training from scratch.",
    #     choices=MODEL_TYPES,
    # )
    args.push_to_hub = None
    # parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    args.hub_model_id = None
    # parser.add_argument(
    #     "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    # )
    args.hub_token = None
    # parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    # args = parser.parse_args()
    #
    # # Sanity checks
    #
    # if args.dataset_name is None and args.train_file is None and args.validation_file is None:
    #     raise ValueError("Need either a task name or a training/validation file.")
    #
    # if args.train_file is not None:
    #     extension = args.train_file.split(".")[-1]
    #     assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    # if args.validation_file is not None:
    #     extension = args.validation_file.split(".")[-1]
    #     assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    #
    # if args.push_to_hub:
    #     assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def get_translation_FIM(args, model, tokenizer, layer_name, logger):
    # Parse the arguments
    args = parse_args()
    args.per_device_train_batch_size = 128

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # if args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    # elif args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    # else:
    #     raise ValueError(
    #         "You are instantiating a new tokenizer from scratch. This is not supported by this script."
    #         "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    #     )

    # if args.model_name_or_path:
    #     model = AutoModelForSeq2SeqLM.from_pretrained(
    #         args.model_name_or_path,
    #         from_tf=bool(".ckpt" in args.model_name_or_path),
    #         config=config,
    #     )
    # else:
    #     logger.info("Training new model from scratch")
    #     model = AutoModelForSeq2SeqLM.from_config(config)
    #
    # model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        assert (
            args.target_lang is not None and args.source_lang is not None
        ), "mBart requires --target_lang and --source_lang"
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if args.source_lang is not None:
            tokenizer.src_lang = args.source_lang
        if args.target_lang is not None:
            tokenizer.tgt_lang = args.target_lang

    # Get the language codes for input/target.
    source_lang = args.source_lang.split("_")[0]
    target_lang = args.target_lang.split("_")[0]

    padding = "max_length" if args.pad_to_max_length else False

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, train_dataloader = accelerator.prepare(
        model, train_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    metric = load_metric("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Getting Empirical Fisher Information Matrix *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Layer = {layer_name}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.per_device_train_batch_size), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        FIM = None
        for i in range(args.per_device_train_batch_size):
            input = {}
            for k, v in batch.items():
                input[k] = v[[i]]
            model.zero_grad()
            outputs = model(**input)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            # loss.backward()
            progress_bar.update(1)
            completed_steps += 1
            weight = model.get_submodule(layer_name).weight.grad.data.cpu().numpy().flatten()
            if hasattr(model.get_submodule(layer_name), "bias") and model.get_submodule(layer_name).bias is not None:
                bias = model.get_submodule(layer_name).bias.grad.data.cpu().numpy().flatten()
                param = np.concatenate([weight, bias], axis=0)
            else:
                param = np.concatenate([weight, np.zeros(model.get_submodule(layer_name).weight.data.cpu().numpy().shape[0])], axis=0)
            if FIM is None:
                FIM = param ** 2
            else:
                FIM += param ** 2
        return FIM / args.per_device_train_batch_size


def get_translation_LRC(args, model, tokenizer, logger):
    # Parse the arguments
    args = parse_args()
    args.per_device_train_batch_size = 128

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # if args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    # elif args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    # else:
    #     raise ValueError(
    #         "You are instantiating a new tokenizer from scratch. This is not supported by this script."
    #         "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    #     )

    # if args.model_name_or_path:
    #     model = AutoModelForSeq2SeqLM.from_pretrained(
    #         args.model_name_or_path,
    #         from_tf=bool(".ckpt" in args.model_name_or_path),
    #         config=config,
    #     )
    # else:
    #     logger.info("Training new model from scratch")
    #     model = AutoModelForSeq2SeqLM.from_config(config)
    #
    # model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        assert (
            args.target_lang is not None and args.source_lang is not None
        ), "mBart requires --target_lang and --source_lang"
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if args.source_lang is not None:
            tokenizer.src_lang = args.source_lang
        if args.target_lang is not None:
            tokenizer.tgt_lang = args.target_lang

    # Get the language codes for input/target.
    source_lang = args.source_lang.split("_")[0]
    target_lang = args.target_lang.split("_")[0]

    padding = "max_length" if args.pad_to_max_length else False

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, train_dataloader = accelerator.prepare(
        model, train_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    metric = load_metric("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Getting Empirical Fisher Information Matrix *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.per_device_train_batch_size), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        code_book = []

        def counting_forward_hook(module, inp, out):
            try:
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)

                x = (inp > 0).float()
                code_book.append(x.cpu().numpy())
            except:
                pass

        for name, module in model.named_modules():
            if 'Relu' in str(type(module)):
                # hooks[name] = module.register_forward_hook(counting_hook)
                module.wo.register_forward_hook(counting_forward_hook)

        for i in range(args.per_device_train_batch_size):
            input = {}
            for k, v in batch.items():
                input[k] = v[[i]]
            outputs = model(**input)

            progress_bar.update(1)
            completed_steps += 1
        return np.array(code_book), batch