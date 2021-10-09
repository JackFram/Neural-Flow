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


def eval(model: nn.Module, dataloader):
    print('===== evaluating network ....... =====')
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_num = 0
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            batch_num += 1
            correct += predicted.eq(targets).sum().item()
            # print("{}/{}, {}".format(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)))
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    loss = test_loss/batch_num
    print("{}".format('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (loss, acc, correct, total)))