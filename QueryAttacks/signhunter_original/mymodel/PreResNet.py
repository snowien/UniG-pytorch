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

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
# device = torch.device('cuda:0')

__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

class no_bn(nn.Module):
    def __init__(self):
        super(no_bn, self).__init__()
    def forward(self, x):
        return x


# learnable dimension reduction matrix
class GS_layer(nn.Module):
    def __init__(self, shape):
        super(GS_layer, self).__init__()
        self.shape = shape
        self.p = nn.Parameter(torch.rand(shape).cuda(), requires_grad = True)

    def forward(self, x):
        batch = x.size(0)
        mat, _ = self.p.split([batch, self.p.size(0)-batch], dim=0)
        # print(mat.shape)
        x = x * mat
        # x = x * self.p
        return x

    # reconstruction loss
    def re_loss(self, x, x_true):
        batch = x.size(0)
        mat, _ = self.p.split([batch, self.p.size(0)-batch], dim=0)
        loss = torch.norm(x * mat - x_true)
        # loss = torch.norm(x * self.p - x_true)
        return loss

    def set_param(self):
        torch.nn.init.uniform_(self.p)
        return

# learnable dimension reduction matrix
class flatten(nn.Module):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, w_noise=False):
        super(PreActBlock, self).__init__()
        if not w_noise:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                )
        else:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = noise_Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = noise_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    noise_Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
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

    def __init__(self, in_planes, planes, stride=1, w_noise=False):
        super(PreActBottleneck, self).__init__()
        if not w_noise:
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
        else:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = noise_Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = noise_Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes)
            self.conv3 = noise_Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    noise_Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
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
    def __init__(self, block, num_blocks, num_classes=100, ifgs=False, train_p_epoch=20, lr_p=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.ifgs = ifgs
        self.train_p_epoch = train_p_epoch
        self.lr = lr_p

        # # train data
        # train_input = np.load('matrix2-clean-x/train.npy')
        # self.train_input = torch.from_numpy(train_input).cuda()
        # self.train_input.requires_grad = True
        # # train label
        # train_label = np.load('matrix2-clean-x/train_label.npy')
        # self.train_label = torch.from_numpy(train_label).cuda()


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

        self.criterion = nn.CrossEntropyLoss()

        self.margin_interval = 3  # 4
        self.reverse_step = 0.2  # 0.1

        self.one_hot = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def optimize(self, x):
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

        feature = out
        out = self.gs(feature)
        out = F.relu(self.bn(out))
        out = self.avgpool(out)
        out = self.flatten(out)

        out = self.linear(out)

        if not self.one_hot:
            return out[0:batch], feature
        else:
            if batch!=1:
                _, pred = out.topk(1, 1, True, True)
                pred = pred.squeeze()
            elif batch==1:
                index = out.argmax()
                pred = torch.tensor([index]).cuda()

            one_hot = torch.zeros(batch, 10).cuda()
            one_hot.scatter_(dim=1, index=pred.unsqueeze(dim=1), src=torch.ones(batch, 10).cuda())
            return one_hot, feature


    def combine(self, x):
        # index = int(44400*torch.rand(size=[1], requires_grad=True))
        index = int(44400 * torch.rand(size=[1]))
        input = torch.cat((x, self.train_input[index:index+10]), 0)
        #train_label = self.train_label[index:index+100]
        return input#, train_label

    def train_grad_simi(self, x):
        input = x.clone().cuda()
        for epoch in range(self.train_p_epoch):
            output, feature = self.optimize(input)
            _, pred = output.topk(1, 1, True, True)
            target = pred.squeeze().detach()
            target = target.view(-1)
            if epoch == 0:
                feature_true = feature.detach()

            loss_ce = self.criterion(output, target)

            self.optimizer.zero_grad()
            grad = torch.autograd.grad(loss_ce, feature, create_graph=True, retain_graph=True)[0]
            if epoch == 0:
                min = grad.min().detach()
                max = grad.max().detach()
            #     print('min:', min, 'max:', max)

            grad = (grad - min) / max - min

            # optimize p
            grad_simi = 0
            for j in range(input.size(0) - 1):
                grad_simi += torch.norm((grad[j] - grad[j + 1]))


            loss_re = self.gs.re_loss(feature, feature_true)
            loss_gs = 1 * grad_simi + 1 * loss_re
            # loss = grad_simi

            self.optimizer.zero_grad()
            loss_gs.backward()
            self.optimizer.step()
            # print('epoch:{:d}, grad_similarity_loss: {:.3f}'.format(epoch, loss.data))
        return


    def forward(self, x):
        batch = x.shape[0]
        x = torch.from_numpy(x).cuda()
        # x, train_label = self.combine(x)
        if self.ifgs:
            self.gs = GS_layer((batch, 512, 4, 4))
            self.optimizer = torch.optim.SGD([self.gs.p], lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.gs.set_param()
            self.train_grad_simi(x)
        else:
            self.gs = no_bn()

        out, _ = self.optimize(x)
        return out



def ResNet18(num_classes = 10, ifgs=False, train_p_epoch=20, lr_p=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes, ifgs, train_p_epoch, lr_p)

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


