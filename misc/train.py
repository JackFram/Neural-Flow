import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
import os
import sys
sys.path.append("../")

from dataset import get_dataset
from model import get_model


parser = argparse.ArgumentParser(description='Network component importance identification')

# Model Args
parser.add_argument('--model', default='resnet18', type=str)
parser.add_argument('--ckpt_dir', default='./checkpoint', type=str, help='pretrained model dir or saving dir')
# Data Args
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--dataroot', default='../data', type=str)
# ML hyper-parameters
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epoch_num', default=200, type=int, help='training epoch')

args = parser.parse_args()
args.model_path = os.path.join(args.ckpt_dir, "ckpt_{}_{}.pth".format(args.model, args.dataset))


def train(args):
    data = get_dataset(args.dataset, args.dataroot)
    trainloader = data.get_train_loader()
    testloader = data.get_test_loader()
    net = get_model(args.model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.ckpt_dir), 'Error: no checkpoint file found!'
        checkpoint = torch.load(args.model_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

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

        scheduler.step()

if __name__ == '__main__':
    train(args)