from dataset import get_dataset
from tqdm import tqdm
import numpy as np
import copy
import shutil
import time
from enum import Enum
from opt import QuantizeOp, SPruningOp, PruningOp, LowRankOp

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def get_cv_FIM(args, model, tokenizer, layer_name, logger, prefix=""):
    # get train loader
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    ds = get_dataset(args.data, args)
    train_loader = ds.get_train_loader()
    print("***** Getting Empirical Fisher Information Matrix *****")
    print(f"  Layer = {layer_name}")

    progress_bar = tqdm(range(args.train_batch_size))
    model.train()
    for _, batch in enumerate(train_loader):
        FIM = None
        for i in range(args.train_batch_size):
            image, target = batch[0][i].unsqueeze(0), batch[1][i].unsqueeze(0)
            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            progress_bar.update(1)
            weight = model.get_submodule(layer_name).weight.grad.cpu().numpy().flatten()
            if hasattr(model.get_submodule(layer_name), "bias") and model.get_submodule(layer_name).bias is not None:
                bias = model.get_submodule(layer_name).bias.grad.cpu().numpy().flatten()
                param = np.concatenate([weight, bias], axis=0)
            else:
                param = np.concatenate([weight, np.zeros(model.get_submodule(layer_name).weight.grad.cpu().numpy().shape[0])], axis=0)
            if FIM is None:
                FIM = param**2
            else:
                FIM += param**2
        return FIM/args.train_batch_size

def remove_prune(model, name_list):
    op = SPruningOp(model)
    for name in name_list:
        if name + ".SVDLinear-0" in op.operatable:
            mod_1 = model.get_submodule(name + ".SVDLinear-0")
            mod_2 = model.get_submodule(name + ".SVDLinear-1")
            prune.remove(mod_1, "weight")
            prune.remove(mod_2, "weight")

        if name + ".SVDConv-0" in op.operatable:
            mod_1 = model.get_submodule(name + ".SVDConv-0")
            mod_2 = model.get_submodule(name + ".SVDConv-1")
            prune.remove(mod_1, "weight")
            prune.remove(mod_2, "weight")

        if name in op.operatable:
            mod = model.get_submodule(name)
            prune.remove(mod, "weight")

def evaluate_cv_solver(solver, get_solution_func, model_orig, args, **kwargs):
    loss = []
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model_orig.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    ds = get_dataset(args.data, args)
    train_loader = ds.get_train_loader()
    val_loader = ds.get_test_loader()
    for storage_thresold in np.arange(solver.model_size-solver.model_size/20, 0, -solver.model_size/20):
        model = copy.deepcopy(model_orig)
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        prune_set = set()

        print("Getting results for storage threshold {}".format(storage_thresold))
        if 'methods' in kwargs:
            solution = get_solution_func(storage_thresold, methods=kwargs['methods'])
        else:
            solution = get_solution_func(storage_thresold)
        print(f"solution: {solution}")
        if solution is not None:
            quantize_list = []
            for layer in solution:
                for name in layer.split("+"):
                    layer_name, op_name, attrs = name.split("@")
                    if op_name == "upruning":
                        if float(attrs) >= 0:
                            prune_set.add(layer_name)
                        op = PruningOp(model)
                        model = op.apply([layer_name], amount=float(attrs), inplace=True)
                    elif op_name == "quantize" and attrs != "none":
                        quantize_list.append(layer_name)
                    elif op_name == "lowrank":
                        op = LowRankOp(model)
                        model = op.apply([layer_name], rank_fraction=(float(attrs)), inplace=True)
                    elif op_name == "spruning":
                        if float(attrs) >= 0:
                            prune_set.add(layer_name)
                        op = SPruningOp(model)
                        model = op.apply([layer_name], amount=float(attrs), inplace=True)
            results = train(
                train_loader=train_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=1,
                args=args,
            )
            if len(prune_set) != 0:
                remove_prune(model, list(prune_set))
            if len(quantize_list) > 0:
                op = QuantizeOp(model)
                op.set_config()
                mod_model = op.apply(name_list=quantize_list, verbose=False, inplace=True)
            else:
                mod_model = model
            # training_args.no_cuda = True
            results = validate(
                val_loader=val_loader,
                model=mod_model,
                criterion=criterion,
                args=args
            )
            loss.append(results["eval_loss"])
    return loss

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return {"eval_loss":losses.avg, "top1":top1.avg, "top5":top5.avg}


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res