import torchvision
import torchvision.transforms as transforms
import torch
from dataset.imagenet_16 import ImageNet16Data

import os
import argparse


class Cifar10:
    def __init__(self, root_dir="./data"):
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.trainset = torchvision.datasets.CIFAR10(
            root=root_dir, train=True, download=True, transform=self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=128, shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(
            root=root_dir, train=False, download=True, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=100, shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')

    def get_train_loader(self):
        return self.trainloader

    def get_test_loader(self):
        return self.testloader
    
    def get_class_num(self):
        return len(self.classes)


class ImageNet16:
    def __init__(self, root_dir="./data", class_num=120):
        self.class_num = class_num

        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std  = [x / 255 for x in [63.22,  61.26 , 65.09]]

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(16, padding=2),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ])

        self.trainset = ImageNet16Data(
            root=root_dir, train=True, transform=self.transform_train, use_num_of_class_only=class_num)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=256, shuffle=True, num_workers=2)

        self.testset = ImageNet16Data(
            root=root_dir, train=False, transform=self.transform_test, use_num_of_class_only=class_num)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=200, shuffle=False, num_workers=2)

    def get_train_loader(self):
        return self.trainloader

    def get_test_loader(self):
        return self.testloader

    def get_class_num(self):
        return self.class_num