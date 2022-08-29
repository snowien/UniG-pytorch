import os
import time
import torch
import random
import shutil
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
import torchvision
from torchvision import transforms

import torchvision.utils as vutil
import math
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt 
import PIL.Image


__all__ = ['save_checkpoint', 'setup_dataset_models', 'setup_seed', 'print_args',
           'test', 'test_adv', 'plot_feature', 'test_ece', 'plot_ece_figure']

def dense_to_onehot(y_test, n_cls):
    y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
    y_test_onehot[np.arange(len(y_test)), y_test] = True
    return y_test_onehot

def load_cifar10(n_ex, train=False):
    testset = torchvision.datasets.CIFAR10(root='data', train=train, download=True)
    data = np.transpose(testset.data.astype(np.float32), axes=[0, 3, 1, 2]) / 255.0
    label = np.array(testset.targets)
    return data[:n_ex], label[:n_ex]

def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_feature(feature):
    shape = feature.shape
    batch = shape[0]
    channel = shape[1]
    for i in range(batch):
        os.makedirs('feature/' + str(i) + '/', exist_ok=True)
        data = feature[i]
        for j in range(channel):
            save = data[j].detach().cpu().numpy()
            plt.imsave('feature/' + str(i) + '/' + str(j) + '.png', save)
    return

def plot_input(x):
    shape = x.shape
    batch = shape[0]
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    for i in range(batch):
        os.makedirs('feature/', exist_ok=True)
        pic = toPIL(x[i].detach().cpu())
        pic.save('feature/' + str(i) + '.png')
    return

def batch_grad_consistency(input_adv, input):
    shape = input.shape
    batch = shape[0]
    noise = input_adv-input
    grad_cons = 0
    for j in range(batch - 1):
        grad_cons += torch.norm((noise[j]-noise[j+1]))
    return grad_cons
# testing
def test(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    losses_pca = AverageMeter()
    model.eval()
    start = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # comput output
        output = model(input)
        loss = criterion(output, target)

        loss = loss.float()
        # loss_pca = loss_pca.float()
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        # losses_pca.update(loss_pca.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})   \t'
                  # 'Loss_var {losses_var.val:.4f} ({losses_var.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Time {2:.2f}'.format(
                i, len(val_loader), end - start, losses=losses, losses_var=losses_pca, top1=top1))
        # print('Standard Accuracy {top1.avg:.3f}'.format(top1=top1))

    print('Standard Accuracy {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

# def test_ece(label_path='clean_TestLabel.npy', pred_path='clean_TestPred.npy'):
def test_ece(val_loader, model):
    model.eval()
    pred = []
    label = []
    for i, (input, target) in enumerate(val_loader):
        ### label ###
        target = target.unsqueeze(1)
        target = torch.zeros(target.size(0), 10).scatter_(1, target, 1)
        for j in range(target.size(0)):
            label.append(target[j].cpu().numpy())
        ### prediction ###
        input = input.cuda()
        output = model(input)
        output = output.detach().cpu().numpy()
        output = softmax(output, axis=1)
        for j in range(target.size(0)):
            pred.append(output[j])

    pred = np.array(pred)
    label = np.array(label)

    ece = ece_score(pred, label, n_bins=15)
    print(ece)
    return ece


def ece_score(y_pred, y_test, n_bins=15):
    # t = np.exp(z)
    # a = np.exp(y_pred) / np.sum(y_pred, axis=1)
    # print(ece_score(, y_test, n_bins=15))

    # py = softmax(y_pred, axis=1)
    # py = np.array(py)
    # y_test = np.array(y_test)
    py = y_pred
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)


# def plot_ece_figure(aaa_model, args):
#     plt.rcParams["figure.figsize"] = (20.0, 5.0)
#     plt.rcParams["figure.dpi"] = 500

#     n_samples = 10000
#     n_division = 100
#     n_sample_per_division = int(n_samples / n_division)

#     probs = np.load('clean_TestPred-AdvNet.npy')
#     y_test = np.load('clean_TestLabel.npy')
#     pred = np.load('clean_TestPredGS-AdvNet.npy')
#     prob_ori_index = np.argsort((probs * y_test).max(1))[::-1]
#     colors = ['orange', 'seagreen', 'purple', 'blue', 'green']

#     def plot_grid(probs, label, color_index):
#         prob_mean = []
#         prob_max = (probs * y_test).max(1)
#         ece = ece_score(probs, y_test)
#         for i in range(n_division):
#             prob_mean.append(
#                 np.mean(prob_max[prob_ori_index[i * n_sample_per_division: (i + 1) * n_sample_per_division]]))
#         plt.plot(prob_mean[::-1], label='Confidence of ' + label + '   ECE %.2f' % (ece * 100),
#                  color=colors[color_index])

#         prob_mean = []
#         corr = probs.argmax(1) == y_test.argmax(1)
#         acc = corr.mean()

#         for i in range(n_division):
#             prob_mean.append(np.mean(corr[prob_ori_index[i * n_sample_per_division: (i + 1) * n_sample_per_division]]))
#         plt.plot(prob_mean[::-1], label='Accuracy of ' + label + '   Acc %.2f' % (acc * 100), color=colors[color_index],
#                  linestyle='dotted')
#         return acc, ece

#     for i, margin_interval in enumerate([1, 2, 3, 4, 5, 6]):
#         for j, reverse_step in enumerate([0.05, 0.1, 0.15, 0.2, 0.25, 0.3]):
#             aaa_model.margin_interval = margin_interval
#             aaa_model.reverse_step = reverse_step
#             acc, ece = plot_grid(probs, 'None', 0)
#             if not (i + j): print('hpGS/%s_Acc%.2f_ECE%.2f' % (args.arch, acc * 100, ece * 100))
#             acc, ece = plot_grid(pred, args.arch, 1)
#             plt.ylim(-0.05, 1.05)
#             plt.legend(loc='lower right')
#             result_path = 'hpGS/%s_Acc%.2f_ECE%.2f_MgIv%d_LR%.2f.png' % (
#             args.arch, acc * 100, ece * 100, margin_interval, reverse_step)
#             print(result_path)
#             plt.savefig(result_path)
#             plt.close()
#     return

def load_imagenet(n_ex, model, seed=0):
    try: 
        arch = model.arch_ori
        assert os.path.exists('data/imagenet_%s_imgs_%d.npy' % (arch, seed))
    except AttributeError: arch = model.arch
    data_path = 'data/imagenet_%s_imgs_%d.npy' % (arch, seed)
    label_path = 'data/imagenet_%s_lbls_%d.npy' % (arch, seed)
    # print(data_path, label_path)
    if not os.path.exists(data_path) or not os.path.exists(label_path):
        with open('data/val.txt', 'r') as f: txt = f.read().split('\n')
        labels = {}
        for item in txt:
            if ' ' not in item: continue
            file, cls = item.split(' ')
            labels[file] = int(cls)
        
        data = []
        files = os.listdir('data/ILSVRC2012_img_val')
        label = np.zeros((min([1000, n_ex]), 1000), dtype=np.uint8)
        label_done = []
        random.seed(seed)
        
        for i in random.sample(range(len(files)), len(files)):
            file = files[i]
            lbl = labels[file]
            if lbl in label_done: continue
            
            img = np.array(PIL.Image.open(
                'data/ILSVRC2012_img_val' + '/' + file).convert('RGB').resize((224, 224))) \
                .astype(np.float32).transpose((2, 0, 1)) / 255
            prd = model(img[np.newaxis, ...]).argmax(1)
            if prd != lbl: continue
            
            label[len(data), lbl] = 1
            data.append(img)
            label_done.append(lbl)
            print('selecting samples in different classes...', len(label_done), '/',1000, end='\r')
            if len(label_done) == min([1000, n_ex]): break

        x_test = np.array(data)
        y_test = np.array(label)
        np.save(data_path, x_test)
        np.save(label_path, y_test)
    else:
        x_test = np.load(data_path)
        y_test = np.load(label_path)
    return x_test[:n_ex], y_test[:n_ex]

def ece_score(y_pred, y_test, n_bins=15):
    py = softmax(y_pred, axis=1) if y_pred.max() > 1 else y_pred

    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)