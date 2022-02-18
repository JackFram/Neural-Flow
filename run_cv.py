import argparse
import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.models as models

from dataset import get_dataset
from model import ResNet18
from misc.cv_utils import *
from solver import OneShotHessianSolver
from opt import QuantizeOp, SPruningOp, PruningOp, LowRankOp
import matplotlib.pylab as plt
import pickle
import copy

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('data', type=str,
                    help='dataset name')
parser.add_argument('task', type=str, help='task name')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--train-batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--test-batch-size', default=200, type=int,
                    metavar='N',
                    help='mini-batch size for test')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--pretrained-weights', default='', type=str, metavar='PATH',
                    help='path to custom state dicts (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # Simply call main_worker function
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.pretrained_weights:
        print("=> using custom pre-trained model from '{}'".format(args.pretrained_weights))
        if 'cifar10' in args.pretrained_weights:
            model = ResNet18(10)
            checkpoint = torch.load(args.pretrained_weights)
            model.load_state_dict(checkpoint['net'])
        else:
            print("invalid model creation!!")
            exit(0)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         if args.gpu is None:
    #             checkpoint = torch.load(args.resume)
    #         else:
    #             # Map model to be loaded to specified single gpu.
    #             loc = 'cuda:{}'.format(args.gpu)
    #             checkpoint = torch.load(args.resume, map_location=loc)
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         if args.gpu is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(args.gpu)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    Ops = [QuantizeOp, PruningOp, SPruningOp]
    hession_solver = OneShotHessianSolver(model.eval(), Ops, args, task_name=args.task)

    pqd_loss = evaluate_cv_solver(
        solver=hession_solver,
        get_solution_func=hession_solver.get_filtered_solution,
        model_orig=model,
        args=args,
        methods={"upruning", "spruning", "quantize"}
    )
    # for epoch in range(args.start_epoch, args.epochs):
    #     adjust_learning_rate(optimizer, epoch, args)

    #     # train for one epoch
    #     train(train_loader, model, criterion, optimizer, epoch, args)

    #     # evaluate on validation set
    #     acc1 = validate(val_loader, model, criterion, args)

    #     # remember best acc@1 and save checkpoint
    #     is_best = acc1 > best_acc1
    #     best_acc1 = max(acc1, best_acc1)

        
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'arch': args.arch,
    #         'state_dict': model.state_dict(),
    #         'best_acc1': best_acc1,
    #         'optimizer' : optimizer.state_dict(),
    #     }, is_best)

    np.savez(f"./results/data/{args.task}.npz", pqd_loss=pqd_loss)

    oshs_range = np.arange(hession_solver.model_size, 0, -hession_solver.model_size / 20)
    
    plt.plot(oshs_range, pqd_loss, label="pqd_loss")

    plt.legend()

    plt.savefig(f"./results/graph/{args.task}.pdf", bbox_inches="tight", dpi=500)

if __name__ == '__main__':
    main()

