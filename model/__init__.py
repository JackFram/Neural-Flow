from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .regnet import *
from .dla_simple import *
from .dla import *


def get_model(name: str, num_classes: int) -> nn.Module:
    print('===== {} model loading.. ====='.format(name))
    if name == 'vgg':
        return VGG('VGG19')
    elif name == 'resnet18':
        return ResNet18(num_classes)
    elif name == 'mobile_net':
        return MobileNet()
