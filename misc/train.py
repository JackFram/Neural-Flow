import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import time
import copy
import numpy as np

sys.path.append("../")

from dataset import get_dataset
from model import get_model
from netflow import *
from metric import TopologySimilarity
import seaborn as sns
import matplotlib.pylab as plt

parser = argparse.ArgumentParser(description='Network component importance identification')

# Model Args
parser.add_argument('--model', default='resnet18', type=str)
parser.add_argument('--ckpt_dir', default='../../data/checkpoint', type=str, help='pretrained model dir or saving dir')
# Data Args
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--dataroot', default='../../data', type=str)
# ML hyper-parameters
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epoch_num', default=200, type=int, help='training epoch')
parser.add_argument('--plot', default=False, help='plot score transformation along training')
parser.add_argument('--plot_step', default=10, help='step size for plotting score transformation along training')

args = parser.parse_args()
args.model_path = os.path.join(args.ckpt_dir, "ckpt_{}_{}.pth".format(args.model, args.dataset))

if __name__ != '__main__':
    args.dataroot = '../data'

data = get_dataset(args.dataset, args.dataroot)
trainloader = data.get_train_loader()
testloader = data.get_test_loader()
dataloaders = {"train": trainloader, "val": testloader}


def train(args):
    net = get_model(args.model, num_classes=data.get_class_num())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    net.to(device)

    #### plot ####
    # print("Generating plot")
    #
    # flow = FxInt(net)
    #
    # x, y = next(iter(trainloader))
    # x = x.to(device)
    # flow.run(x)
    # feature_list = flow.get_feature_list()
    # name_list = flow.get_name_list()
    #
    # metric = TopologySimilarity()
    #
    # ret = metric.get_all_layer_batch_score(feature_list)
    #
    # labels = name_list
    #
    # idx_list = np.arange(len(name_list))
    #
    # ax = sns.heatmap(ret, linewidth=0.5)
    # ax.set_xticks(idx_list)
    # ax.set_yticks(idx_list)
    # ax.set_xticklabels(labels, fontsize=5)
    # ax.set_yticklabels(labels, fontsize=5)
    # plt.savefig("../results/{}_{}_topo_score.pdf".format(args.model, 0), bbox_inches="tight", dpi=500)
    # plt.clf()
    #### plot ####

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.ckpt_dir), 'Error: no checkpoint file found!'
        checkpoint = torch.load(args.model_path)
        # net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + args.epoch_num):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print("{}/{}, {}".format(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total)))

        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                print("{}/{}, {}".format(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total)))
                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(args.ckpt_dir):
                os.mkdir(args.ckpt_dir)
            torch.save(state, args.model_path)
            best_acc = acc

        if args.plot and epoch % args.plot_step == 0:
            print("Generating plot")
            flow = FxInt(net)

            x, y = next(iter(trainloader))
            x = x.to(device)
            flow.run(x)
            feature_list = flow.get_feature_list()
            name_list = flow.get_name_list()

            metric = TopologySimilarity()

            ret = metric.get_all_layer_batch_score(feature_list)

            labels = name_list

            idx_list = np.arange(len(name_list))

            ax = sns.heatmap(ret, linewidth=0.5)
            ax.set_xticks(idx_list)
            ax.set_yticks(idx_list)
            ax.set_xticklabels(labels, fontsize=5)
            ax.set_yticklabels(labels, fontsize=5)
            plt.savefig("../results/{}_{}_topo_score.pdf".format(args.model, epoch + 1), bbox_inches="tight", dpi=500)
            plt.clf()

        scheduler.step()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    """
    Support function for model training.

    Args:
      model: Model to be trained
      criterion: Optimization criterion (loss)
      optimizer: Optimizer to use for training
      scheduler: Instance of ``torch.optim.lr_scheduler``
      num_epochs: Number of epochs
      device: Device to run the training on. Must be 'cpu' or 'cuda'
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            count = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                count += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / count
            epoch_acc = running_corrects.double() / count
            print(count, len(dataloaders[phase]))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    train(args)
