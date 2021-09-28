import torch
import torch.nn as nn
import torch.nn.functional as F
# from netflow.net_interpreter import NetIntBase


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        feature_list = {"path":[], "shortcut": []}
        name_list = {"path":[], "shortcut":[]}
        out = self.conv1(x)
        name_list["path"].append(self.conv1.__class__.__name__)
        feature_list["path"].append(out)
        out = self.bn1(out)
        name_list["path"].append(self.bn1.__class__.__name__)
        feature_list["path"].append(out)
        out = F.relu(out)
        name_list["path"].append("ReLU")
        feature_list["path"].append(out)
        out = self.conv2(out)
        name_list["path"].append(self.conv2.__class__.__name__)
        feature_list["path"].append(out)
        out = self.bn2(out)
        name_list["path"].append(self.bn2.__class__.__name__)
        feature_list["path"].append(out)
        for layer in self.shortcut:
            x = layer(x)
            name_list["shortcut"].append(layer.__class__.__name__)
            feature_list["shortcut"].append(x)
        out += x
        name_list["path"].append("merge shortcut")
        feature_list["path"].append(out)
        out = F.relu(out)
        name_list["path"].append("ReLU")
        feature_list["path"].append(out)
        return {"BasicBlock": feature_list}, {"BasicBlock": name_list}, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        feature_list = {"path": [], "shortcut": []}
        name_list = {"path": [], "shortcut": []}

        out = self.conv1(x)
        name_list["path"].append(self.conv1.__class__.__name__)
        feature_list["path"].append(out)

        out = self.bn1(out)
        name_list["path"].append(self.bn1.__class__.__name__)
        feature_list["path"].append(out)

        out = F.relu(out)
        name_list["path"].append("ReLU")
        feature_list["path"].append(out)

        out = self.conv2(out)
        name_list["path"].append(self.conv2.__class__.__name__)
        feature_list["path"].append(out)

        out = self.bn2(out)
        name_list["path"].append(self.bn2.__class__.__name__)
        feature_list["path"].append(out)

        out = F.relu(out)
        name_list["path"].append("ReLU")
        feature_list["path"].append(out)

        out = self.conv3(out)
        name_list["path"].append(self.conv3.__class__.__name__)
        feature_list["path"].append(out)

        out = self.bn3(out)
        name_list["path"].append(self.bn3.__class__.__name__)
        feature_list["path"].append(out)

        for layer in self.shortcut:
            x = layer(x)
            name_list["shortcut"].append(layer.__class__.__name__)
            feature_list["shortcut"].append(x)
            print(name_list)

        out += x
        name_list["path"].append("merge shortcut")
        feature_list["path"].append(out)

        out = F.relu(out)
        name_list["path"].append("ReLU")
        feature_list["path"].append(out)
        return {"Bottleneck": feature_list}, {"Bottleneck": name_list}, out


class ResNetInt(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetInt, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = {"path": []}
        name_list = {"path": []}

        out = self.conv1(x)
        name_list["path"].append(self.conv1.__class__.__name__)
        feature_list["path"].append(out)

        out = self.bn1(out)
        name_list["path"].append(self.bn1.__class__.__name__)
        feature_list["path"].append(out)

        out = F.relu(out)
        name_list["path"].append("ReLU")
        feature_list["path"].append(out)

        for layer in self.layer1:
            feature_dict, name_dict, out = layer(out)
            for key in feature_dict:
                name_list["path"].append({key: name_dict[key]})
                feature_list["path"].append({key: feature_dict[key]})

        for layer in self.layer2:
            feature_dict, name_dict, out = layer(out)
            for key in feature_dict:
                name_list["path"].append({key: name_dict[key]})
                feature_list["path"].append({key: feature_dict[key]})

        for layer in self.layer3:
            feature_dict, name_dict, out = layer(out)
            for key in feature_dict:
                name_list["path"].append({key: name_dict[key]})
                feature_list["path"].append({key: feature_dict[key]})

        for layer in self.layer4:
            feature_dict, name_dict, out = layer(out)
            for key in feature_dict:
                name_list["path"].append({key: name_dict[key]})
                feature_list["path"].append({key: feature_dict[key]})

        out = F.avg_pool2d(out, 4)
        name_list["path"].append("AvgPool2d_4")
        feature_list["path"].append(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return feature_list, name_list, out


def ResNetInt18():
    return ResNetInt(BasicBlock, [2, 2, 2, 2])


def ResNetInt34():
    return ResNetInt(BasicBlock, [3, 4, 6, 3])


def ResNetInt50():
    return ResNetInt(Bottleneck, [3, 4, 6, 3])


def ResNetInt101():
    return ResNetInt(Bottleneck, [3, 4, 23, 3])


def ResNetInt152():
    return ResNetInt(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNetInt18()
    feature_list, name_list, y = net(torch.randn(1, 3, 32, 32))
    print(y.size(), name_list["path"][3]["BasicBlock"], feature_list["path"][3]["BasicBlock"]["path"][0].shape)

if __name__ == "__main__":
    test()

# class ResNetInt(NetIntBase):
