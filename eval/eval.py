from netflow.vgg_interpreter import VggInt
from dataset.cv_dataloader import Cifar10
from model import *
from netflow import *
from utils import *

import torch.nn as nn
import torch.optim as optim


def train(data_loader, model, criterion, optimizer, epochs=50, print_every=10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
    
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()
    model.to(device)
    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % print_every == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, epochs, i+1, total_step, loss.item()))

def interpret_model(data_loader, model, model_interpreter):
    train_loader = data_loader.get_train_loader()

    state_dict = model.state_dict()
    model_interpreter.load_state_dict(state_dict)
    
    train_batch = next(iter(train_loader))[0]

    return model_interpreter(train_batch)

def main():
    data_loader = Cifar10()
    model = ResNet18()
    interpreter = ResNetInt18()
    # model = VGG('VGG11')
    # interpreter = VggInt(model)

    learning_rate = 0.001
    epoch = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(data_loader, model, criterion, optimizer, epoch, 1)

    result = interpret_model(data_loader, model, interpreter)

    print(result[0])

if __name__ == "__main__":
    main()