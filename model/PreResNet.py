'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from advertorch.utils import NormalizeByChannelMeanStd
import numpy as np
import time

__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

class no_bn(nn.Module):
    def __init__(self):
        super(no_bn, self).__init__()
    def forward(self, x):
        return x


class P_layer(nn.Module):
    def __init__(self, shape):
        super(P_layer, self).__init__()
        self.shape = shape
        self.p = nn.Parameter(torch.rand(shape).cuda(), requires_grad = True)
        
    def forward(self, x):
        batch = x.size(0)
        mat, _ = self.p.split([batch, self.p.size(0)-batch], dim=0)
        x = x * mat
        return x

    def init_param(self):
        # self.p.data = self.save_p
        # torch.nn.init.constant_(self.p, 1)
        # torch.nn.init.uniform_(self.p)
        torch.nn.init.normal_(self.p, mean=1, std=0.5)
        return
       
class flatten(nn.Module):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        
        # default normalization is for CIFAR10
        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = flatten()

        self.gs = no_bn()
        self.onlyout = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_undefend(self, x):
        x = self.normalize(x)
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.relu(self.bn(out))
        out = self.avgpool(out)
        out = self.flatten(out)

        fea = out
        out = self.linear(out)
        if self.onlyout == False:
            return out, fea
        elif self.onlyout == True:
            return out

    def get_feature(self, x):
        with torch.no_grad():
            # forward
            x = self.normalize(x)
            out = self.conv1(x)

            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            out = F.relu(self.bn(out))
            out = self.avgpool(out)
            out = self.flatten(out)
            return out
    
    def get_pred(self, x):
        fea = x
        out = self.gs(fea)
        out = self.linear(out)
        return out
        
    def undefend_get_pred(self, x):
        out = self.linear(x)
        return out
    
    def forward(self, x):
        batch = x.size(0)
        # 级联[train-data, test-data]
        # x = self.combine(x)
        # forward
        x = self.normalize(x)
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.relu(self.bn(out))
        out = self.avgpool(out)
        out = self.flatten(out)
        
        fea = out
        out = self.gs(fea)
        after = out

        out = self.linear(out)
        if self.onlyout == False:
            return out[0:batch], fea, after
        elif self.onlyout == True:
            return out[0:batch]

    def set_gs_param(self, shape):
        self.gs = P_layer(shape)
        self.gs.init_param()
        return
    
    def set_gs_none(self):
        self.gs = no_bn()
        return

    def set_onlyout(self, flag):
        self.onlyout = flag
        return

    def linear_layer(self, x):
        out = self.linear(x)
        return out


def ResNet18(num_classes = 10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes = 10):
    return PreActResNet(PreActBlock, [3,4,6,3], num_classes)

def ResNet50(num_classes = 10):
    return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes)

def ResNet101(num_classes = 10):
    return PreActResNet(PreActBottleneck, [3,4,23,3], num_classes)

def ResNet152(num_classes = 10):
    return PreActResNet(PreActBottleneck, [3,8,36,3], num_classes)


# m = self.gs.p.clone()
# m = m.view(m.size(0), -1)
# m = m.cpu().detach().numpy()
# print(np.linalg.matrix_rank(m))
# print(afdf)


