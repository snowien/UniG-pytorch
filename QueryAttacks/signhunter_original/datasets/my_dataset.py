"""
A wrapper for datasets
    mnist, cifar10, imagenet
"""
import os

import numpy as np
import torch 
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Dataset(object):
    def __init__(self, name, config):
        """
        :param name: dataset name
        :param config: dictionary whose keys are dependent on the dataset created
         (see source code below)
        """
        assert name in ['mnist', 'cifar10', 'cifar10aug', 'imagenet'], "Invalid dataset"
        self.name = name

        if self.name == 'cifar10':
            self.data = datasets.CIFAR10(root='./data/cifar10', train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor()
            ]))
        elif self.name == 'cifar100':
            self.data = datasets.CIFAR100(root='../data', train=False, download=True, 
                transform=transforms.Compose([transforms.ToTensor()
            ]))
        elif self.name == 'mnist':
            self.data = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
        elif self.name == 'imagenet':
            valdir = './data/ImageNet2012/ILSVRC2012_img_val'
            # valdir = os.path.join('/home/datasets/ILSVRC2012/', 'val')
            # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                             std=[0.229, 0.224, 0.225])
            self.data = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
            ]))

    def get_eval_data(self, bstart, bend):
        """
        :param bstart: batch start index
        :param bend: batch end index
        """
        if self.name in ['cifar10', 'cifar100', 'mnist']:
            return self.data.data[bstart:bend, :], \
                   self.data.targets[bstart:bend]
        elif self.name == 'imagenet':
            input = []
            label = []
            for i in range(bstart, bend):
                x, y = self.data[i][0], self.data[i][1]
                x = x.cpu().numpy()
                input.append(x)
                label.append(y)
            input = np.array(input)
            label = np.array(label)
            # print(input.shape, label.shape)
            return input, label
        else:
            raise NotImplementedError

    @property
    def min_value(self):
        if self.name in ['cifar10', 'cifar100', 'mnist', 'imagenet']:
            return 0

    @property
    def max_value(self):
        if self.name in ['cifar10', 'cifar100']:
            return 255
        if self.name in ['mnist', 'imagenet']:
            return 1
