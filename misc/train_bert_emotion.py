
from __future__ import absolute_import, division, print_function

import numpy as np
import os
import torch
import time
import math

from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import (DataLoader, SequentialSampler, DistributedSampler,
                              TensorDataset)
from tqdm import tqdm
import logging
from accelerate import Accelerator
from transformers import glue_compute_metrics as compute_metrics
import transformers

from transformers import (
    AdamW,
    get_scheduler,
)
from datasets import load_dataset

def evaluate(args, model, tokenizer, logger, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    accelerator = Accelerator()

    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_emotion_dataset(args, logger, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.device == "gpu":

            model, eval_dataloader = accelerator.prepare(
                model, eval_dataloader
            )

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            # batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = out_label_ids.flatten()
        f1 = f1_score(labels_flat, preds_flat, average="weighted")
        acc = accuracy_score(labels_flat, preds_flat)
        acc_f1 = (f1 + acc)/2
        result = {
            "f1": f1,
            "acc": acc,
            "acc_and_f1": acc_f1
        }
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def load_emotion_dataset(args, logger, tokenizer, evaluate):
    cached_encoded_data = os.path.join(args.data_dir, f'cached_emotion_{"val" if evaluate else "train"}')
    if os.path.exists(cached_encoded_data) and not args.overwrite_cache:
        logger.info("Loading encoded data from cached file %s", cached_encoded_data)
        encoded_data = torch.load(cached_encoded_data)
    else:
        logger.info("Creating encoded data from dataset file")
        data = load_dataset("emotion")
        if evaluate:
            data = data["validation"].to_pandas()
        else:
            data = data["train"].to_pandas()

        encoded_data = tokenizer.batch_encode_plus(
            data.text.values,
            add_special_tokens=True,
            return_attention_mask=True,
            padding="max_length",
            return_tensors='pt'
        )
        encoded_data["labels"] = torch.tensor(data.label.values, dtype=torch.long)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_encoded_data)
            torch.save(encoded_data, cached_encoded_data)

    all_input_ids = encoded_data["input_ids"]
    all_attention_mask = encoded_data["attention_mask"]
    all_token_type_ids = encoded_data["token_type_ids"]
    all_labels = encoded_data["labels"]

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def train_bert(args, model, tokenizer, logger, prefix=""):
    accelerator = Accelerator()
    logger.info(accelerator.state)

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    train_dataset = load_emotion_dataset(args, logger, tokenizer, evaluate=False)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

     # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

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

    if args.device == "gpu":

        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
        results = evaluate(args, model, tokenizer, logger, prefix)

    return results

def get_bert_FIM(args, model, tokenizer, layer_name, logger, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    accelerator = Accelerator()
    logger.info(accelerator.state)

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    train_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    train_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for train_task, train_output_dir in zip(train_task_names, train_outputs_dirs):
        train_dataset = load_emotion_dataset(args, logger, tokenizer, evaluate=False)

        args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        model, train_dataloader = accelerator.prepare(
            model, train_dataloader
        )

        logger.info("***** Getting Empirical Fisher Information Matrix *****")
        logger.info(f"  Num examples = {args.train_batch_size}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Layer = {layer_name}")

        progress_bar = tqdm(range(args.train_batch_size), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            FIM = None
            for i in range(args.train_batch_size):
                inputs = {
                    'input_ids':      batch[0][[i]],
                    'attention_mask': batch[1][[i]],
                    'labels':         batch[3][[i]]
                }
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2][[i]] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                model.zero_grad()
                outputs = model(**inputs)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                progress_bar.update(1)
                completed_steps += 1
                weight = model.get_submodule(layer_name).weight.data.cpu().numpy().flatten()
                if hasattr(model.get_submodule(layer_name), "bias"):
                    bias = model.get_submodule(layer_name).bias.data.cpu().numpy().flatten()
                    param = np.concatenate([weight, bias], axis=0)
                else:
                    param = weight
                if FIM is None:
                    FIM = param**2
                else:
                    FIM += param**2
            model.to("cpu")
            return FIM/args.train_batch_size

    return results

def time_model_evaluation(model, configs, tokenizer, logger):
    eval_start_time = time.time()
    result = evaluate(configs, model, tokenizer, logger, prefix="")
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    print(result)
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))
    return result