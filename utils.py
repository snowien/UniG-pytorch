import os
import time
import torch
import random
import shutil
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import sys
from advertorch.utils import NormalizeByChannelMeanStd
import json
from torchvision import transforms
from advertorch.attacks import LinfPGDAttack, L2PGDAttack,GradientSignAttack
import math
from torch.autograd import Variable
import torch.optim as optim
from autoattack import AutoAttack
from advertorch.utils import NormalizeByChannelMeanStd
import random
import shutil
import matplotlib.pyplot as plt
from model.PreResNet import ResNet18
####### import attack
import sys
sys.path.append("./QueryAttacks/SimBA_original/simple-blackbox-attack/")
from simba import SimBA
sys.path.append("QueryAttacks/signhunter_original/")
from run_attack import attack_mode
sys.path.append("./QueryAttacks/square_attack/")
from attack import square_attack_linf, square_attack_l2
sys.path.append("./QueryAttacks/blackbox-bandits/src/")
from main_bandits import bandit_attack

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)


# print training configuration
def print_args(args):
    print('*' * 50)
    print('Dataset: {}'.format(args.dataset))
    print('Model: {}'.format(args.arch))
    if args.arch == 'wideresnet':
        print('Depth {}'.format(args.depth_factor))
        print('Width {}'.format(args.width_factor))
    print('*' * 50)
    print('Attack Norm {}'.format(args.norm))
    print('Test Epsilon {}'.format(args.test_eps))
    print('Test Steps {}'.format(args.test_step))
    print('Train Steps Size {}'.format(args.test_gamma))
    print('Test Randinit {}'.format(args.test_randinit))
    if args.eval:
        print('Evaluation')
        print('Loading weight {}'.format(args.pretrained))
    else:
        print('Training')
        print('Train Epsilon {}'.format(args.train_eps))
        print('Train Steps {}'.format(args.train_step))
        print('Train Steps Size {}'.format(args.train_gamma))
        print('Train Randinit {}'.format(args.train_randinit))
        print('SWA={0}, start point={1}, swa_c={2}'.format(args.swa, args.swa_start, args.swa_c_epochs))
        print('LWF={0}, coef_ce={1}, coef_kd1={2}, coef_kd2={3}, start={4}, end={5}'.format(
            args.lwf, args.coef_ce, args.coef_kd1, args.coef_kd2, args.lwf_start, args.lwf_end
        ))


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

def load_cifar10_model(pretrained):
    model = ResNet18(num_classes=10)
    model.normalize = NormalizeByChannelMeanStd(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    print('evaluation from ' + pretrained)
    checkpoint = torch.load(pretrained)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model

def get_output(val_loader, model, model_name):
    model.eval()
    pred = []
    for i, (input, target) in enumerate(val_loader):
        ### prediction ###
        input = input.cuda()
        output = model(input)
        prediction = output.squeeze().detach().cpu().numpy()
        for j in range(input.size(0)):
            pred.append(prediction[j])
        print(i)
    pred = np.array(pred)

    #### saving output
    np.save('./model_output/' + model_name + '.npy', pred)
    return pred

def cal_logit_diff(clean_path, model_path):
    clean_logits = np.load(clean_path)
    model_logits = np.load(model_path)
    print(clean_logits.shape, model_logits.shape)
    diff = []
    avg = 0
    sum = 0
    for i in range(0, clean_logits.shape[0]):
        a = np.linalg.norm((model_logits[i]-clean_logits[i]).flatten(), ord=2)
        sum = sum + a
        avg = sum/(i+1)
        diff.append(a)
    print(i, avg)
    return avg

def cal_flops(model):
    total_num = sum(p.numel() for p in model.parameters())
    print('Total', total_num)
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                             print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return

def get_input_gradient(val_loader, model, model_name):
    model.eval()
    pred = []
    criterion = nn.CrossEntropyLoss()
    for i, (input, target) in enumerate(val_loader):
        ### prediction ###
        input = input.cuda()
        input.requires_grad = True
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        grad = torch.autograd.grad(loss, input, create_graph=True, retain_graph=False)[0]
        # print(grad)
        for j in range(input.size(0)):
            pred.append(grad.data[j].cpu().detach().numpy())
        print(i)
    pred = np.array(pred)
    print(pred.shape)
    #### saving output
    np.save('./model_output/' + model_name + '-grad.npy', pred)
    return pred


########## visualization
def image_filp(image_path):
    import cv2
    img = cv2.imread(image_path)
    # cv2.imshow("yuan", img)
    img1 = cv2.flip(img, 0)  # 镜像
    cv2.imwrite(image_path,img1)
    return

def plot_hyper_param(x, clean, diff, robust, name):
    plt.style.use('seaborn')
    plt.clf()
    x = np.array(x)
    clean = np.array(clean)
    diff = np.array(diff)
    robust = np.array(robust)
    fontsize = 20
    linewidth = 3
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_ylim(0, 100)
    ax1.plot(x, clean, linestyle='-', marker='.', label='CleanAcc',color='#82B0D2',linewidth=linewidth)
    ax1.plot(x, robust, linestyle='-', marker='.', label='AdvAcc',color='#FFBE7A',linewidth=linewidth)
    # ax1.set_xlabel(name, fontsize=fontsize)
    if name=='lr':
        ax1.set_xscale('log', base=10)
    elif name=='batch':
        ax1.set_xscale('log', base=2)
    ax1.set_ylabel('Acc', fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)

    ax2 = ax1.twinx()
    ax2.set_ylim(0, 2)
    ax2.plot(x, diff, linestyle='-', marker='.', label='LogitDiff',color='#8ECFC9',linewidth=linewidth)
    ax2.set_ylabel('LogitDiff', fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.grid()

    fig.legend(loc=1, bbox_to_anchor=(1, 0.35), bbox_transform=ax1.transAxes, fontsize=fontsize)
    plt.savefig('fig/' + name + '.eps', dpi=600, format='eps', bbox_inches='tight')
    plt.savefig('fig/' + name + '.png', dpi=600, format='png', bbox_inches='tight')
    return


def plot_output_diff():
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    plt.clf()
    beta = 0.5
    labels = ['LogitDiff', 'Robustness']
    robust = [0.46, 67.34, 51.22, 77.80]
    logit_diff = [0, 6.994, 1.529, 1.087]
    clean = [94.26, 87.35, 91.14, 94.26]
    x = np.array([0.5, 1, 1.5, 2])*beta
    x_label = ['Vanilla', 'AT', 'RI', 'UniG']

    ### robustness
    bottom = 2
    robust = [i+bottom for i in robust]
    fontsize = 20
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylim(-bottom, 100)
    ax1.set_xlim(0.2*beta, 2.3*beta)
    width = 0.15*beta
    le1 = ax1.bar(x-width, np.array(robust), width=width, bottom=-bottom, align='edge', label=labels[1], color='#FFBE7A')
    ax1.set_xticks(x, labels=x_label, fontsize=fontsize)
    ax1.set_ylabel("Robustness", fontsize=fontsize)
    plt.tick_params(labelsize=20)

    ### logit-diff
    bottom = 0.2
    logit_diff = [i + bottom for i in logit_diff]
    ax2 = ax1.twinx()
    ax2.set_ylim(-bottom, 10)
    le2 = ax2.bar(x, np.array(logit_diff), width=width, bottom=-bottom, align='edge', label=labels[0], color='#8ECFC9')
    # ax2.legend()
    ax2.set_ylabel("LogitDiff", fontsize=fontsize)
    # plt.legend()
    fig.legend(loc=1, bbox_to_anchor=(1,1.03), bbox_transform=ax1.transAxes, fontsize=fontsize)
    plt.grid()
    plt.tick_params(labelsize=20)
    ### legend
    # le = le1+le2
    # labs = [l.get_label() for l in le]
    # ax1.legend(le, labs, loc=0)
    plt.savefig('fig/logit-robust.eps', dpi=600, format='eps', bbox_inches='tight')
    plt.savefig('fig/logit-robust.png', dpi=600, bbox_inches='tight')
    return

def plot_margin_loss():
    van = np.load('square_loss_record/vanilla.npy')
    print(van.shape)
    gsmodel = np.load('square_loss_record/GS.npy')
    gsmodel = gsmodel[0:van.shape[0]]
    atmodel = np.load('square_loss_record/AT.npy')
    atmodel = atmodel[0:van.shape[0]]
    rndmodel = np.load('square_loss_record/RND.npy')
    rndmodel = rndmodel[0:van.shape[0]]
    x = range(van.shape[0])
    plt.clf()
    plt.style.use('seaborn')
    plt.plot(x, van, label='Vanilla', color='#0e72cc')
    plt.plot(x, rndmodel, label='RND', color='#f59311')
    plt.plot(x, atmodel, label='AT', color='#85c021')
    plt.plot(x, gsmodel, label='UniG', color='#fa4343')
    plt.xlabel('Query', fontsize=15)
    plt.ylabel('Margin loss', fontsize=15)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=15, loc='right', bbox_to_anchor=(1, 0.615)) # lower left
    plt.savefig('fig/square.eps', dpi=600, format='eps')#, bbox_inches='tight')
    plt.savefig('fig/square.png', dpi=600)#, bbox_inches='tight')
    return

def plot_feature(feature, name, args):
    # shape = feature.shape
    # batch = shape[0]
    # channel = shape[1]
    num = [22, 26, 33]
    os.makedirs('fig/forward_and_backward/' + name + '/', exist_ok=True)
    for i in num:
        # os.makedirs('feature/' + str(i) + '/', exist_ok=True)
        data = feature[i]
        data = data.reshape(32,64)
        data = (data-data.min())/(data.max()-data.min())
        # data = data * 255
        save = data
        cmap = 'Spectral'
        plt.imsave('fig/forward_and_backward/' + name + '/' + str(i) + '.eps', save, format='eps', dpi=600, cmap=plt.get_cmap(cmap))
        plt.imsave('fig/forward_and_backward/' + name + '/' + str(i) + '.png', save, cmap=plt.get_cmap(cmap))
        # for j in range(channel):
            # save = data[j].detach().cpu().numpy()
            # plt.imsave('feature/' + str(i) + '/' + str(j) + '.png', save)
    return


def plot_input(x, name):
    x = torch.from_numpy(x)
    shape = x.shape
    batch = shape[0]
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    # color_aug = transforms.ColorJitter(brightness=1, contrast=0.5, saturation=0.2, hue=0.1)
    transform = transforms.Compose([
        toPIL,
        # color_aug,
    ])
    ### cifar10
    # num =[7, 24, 26, 34, 37, 46, 53, 57, 63, 66, 68, 70, 74, 81, 86, 87, 91, 97, 101, 116, 125, 127, 129, 134, 139, 145, 148, 153, 160, 169, 180, 183, 187, 192, 197, 201, 204, 206, 213, 221, 237, 247, 255, 263, 264, 266, 269, 273, 276, 279, 284, 287, 302, 307, 308, 309, 313, 314, 323, 332, 342, 343, 345, 351, 352, 366, 368, 369, 375, 376, 388, 397, 398, 405, 411, 424, 426, 430, 437, 439, 446, 449, 454, 456, 459, 463, 464, 470, 473, 477, 478, 483, 485, 488, 491, 497]
    # num = np.load('noise_vis/num.npy')
    ### imagenet
    # num = [13, 17, 19, 22, 26, 33, 39, 41, 46, 48, 52, 63, 64, 65, 85, 87, 91, 93, 95, 98, 109, 111, 117, 130, 132, 136, 140, 162, 163, 166, 169, 172, 174, 182, 187, 204, 213, 221, 226, 252, 254, 258, 267, 274, 278, 293, 296, 301, 305, 307, 308, 316, 318, 322, 324, 329, 341, 344, 379, 383, 385, 393, 399, 402, 410, 415, 416, 425, 431, 432, 438, 445, 450, 451, 454, 463, 466, 471, 474, 475, 476, 483, 494, 496]
    num = [22, 26, 33]
    # for i in range(batch):
    for i in num:
        os.makedirs('fig/forward_and_backward/' + name + '/', exist_ok=True)
        # pic = toPIL(x[i].detach().cpu().view(16,32))
        # pic = toPIL(x[i])
        pic = transform(x[i])
        # if torch.norm(x[i].detach().cpu()) == 0:
        #     continue
        # pic.save('input_vis/' + args.model_type + '-' + str(i) + '.png', quality = 95)
        # num.append(i)
        # print(i)
        pic.save('fig/forward_and_backward/' + name + '/' + str(i) + '.png', quality=95)
    # num = np.array(num)
    print(num)
    # np.save('noise_vis/'+name+'/num.npy', num)
    return


def plot_adv_noise(x, name, args):
    import cv2
    ### imagenet
    os.makedirs('noise_vis/' + args.dataset + '/' + name + '/', exist_ok=True)
    num = [22, 26, 33]
    for i in num:
        a = np.transpose(x[i], [1,2,0])
        a = (a-a.min())/(a.max()-a.min())
        a = a * 255
        # 权重越大，透明度越低
        print(a.dtype)
        a = cv2.addWeighted(a, 0.8, np.zeros(a.shape).astype('float32'), 0, 0)
        print(a.min(), a.max())

        cv2.imwrite('noise_vis/'+ args.dataset + '/' + name+'/%d.png' % i, a)
        # misc.toimage(x[i], cmin=0.0, cmax=...).save('noise_vis/'+ args.dataset + '/' + name+'/' + str(i) + '.png')
        # plt.imsave('noise_vis/'+ args.dataset + '/' + name+'/' + str(i) + '.png', x[i])
    return


def visualize_feature_distribution(fea, target, model, name):
    # model.eval()
    # model.set_onlyout(False)
    # # model.module.set_onlyout(False)
    from sklearn.manifold import TSNE
    # for i, (input, target) in enumerate(val_loader):
    #     input = input.cuda()
    #     target = target.cpu().numpy()
    #     output, fea = model(input)
    #     fea = fea.cpu().detach().numpy()
    print('fea.shape= ', fea.shape)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(fea)
    print('result.shape= ', result.shape)

    def plot_embedding(data, label):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        plt.style.use('seaborn')
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.figure(figsize=(10, 8), dpi=600)
        colors = ['63b2ee', '76da91', 'f8cb7f', 'f89588', '7cd6cf', '9192ab', '7898e1', 'efa666', 'eddd86', '9987ce', '63b2ee', '76da91']
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color='#'+colors[label[i]],
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        # plt.savefig('feature-distribution/'+name+'.png', dpi=600)
        # plt.savefig('feature-distribution/'+name+'.eps', dpi=600, format='eps')
        plt.savefig('fig/'+name+'.png', dpi=600)
        plt.savefig('fig/'+name+'.eps', dpi=600, format='eps')
        return fig
    fig = plot_embedding(result, target)
    # plt.show(fig)
    model.set_onlyout(True)
    # model.module.set_onlyout(True)
    return


# testing
def test(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    start = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        loss = loss.float()
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})   \t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Time {2:.2f}'.format(
                i, len(val_loader), end - start, losses=losses, top1=top1))
            
    end = time.time()
    print('Standard Accuracy {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


#### for square attack
class Model:
    def __init__(self):
        return

    def predict(self, x):
        raise NotImplementedError('use ModelTF or ModelPT')

    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = self.softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

class ModelPT(Model):
    """
    Wrapper class around PyTorch models.

    In order to incorporate a new model, one has to ensure that self.model is a callable object that returns logits,
    and that the preprocessing of the inputs is done correctly (e.g. subtracting the mean and dividing over the
    standard deviation).
    """
    def __init__(self, model):
        super().__init__()
        model.eval()
        self.model = model

    def predict(self, x):
        x = torch.tensor(x).cuda()
        x = x.type(torch.cuda.FloatTensor)
        pred = []
        batch = x.shape[0]
        output = self.model(x)
        prediction = output.detach().cpu().numpy()

        for j in range(batch):
            pred.append(prediction[j])

        pred = np.array(pred)
        return pred

def dense_to_onehot(y_test, n_cls):
    y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
    y_test_onehot[np.arange(len(y_test)), y_test] = True
    return y_test_onehot
#### end for square attack

def test_adv(val_loader, model, criterion, args):
    """
    Run adversarial evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    if args.attack_type == 'square':
        print('eps:{:.3f}, max_query:{:d}, p:{:.3f}'.format(args.test_eps, args.max_query, args.p))
        if args.model_type == 'vanilla' or args.model_type == 'AT':
            version = 'standard'
        else:
            version = 'rand'
        adversary = AutoAttack(model, norm=args.norm, eps=args.test_eps, version=version)
        adversary.attacks_to_run = ['square'] 
        adversary.square.n_queries = args.max_query
        adversary.square.p_init = args.p
    elif args.attack_type == 'simba':
        adversary = SimBA(model, args.dataset, image_size = 224 if args.dataset=='imagenet' else 32)
    elif args.attack_type == 'nes':
        attack_mode(model, args.gpu, batsi=args.batch_size, args = args, cfg= args.dataset + '-nes-linf-config.json')
        return
    elif args.attack_type == 'signhunter':
        attack_mode(model, args.gpu, batsi=args.batch_size, args = args, cfg= args.dataset + '-sign-linf-config.json')
        return
    elif args.attack_type == 'bandits':
        args.json_config = './QueryAttacks-json/' + args.dataset + '-bandits-linf.json'
        bandit_attack(args, model, val_loader)
        return
    elif args.attack_type == 'pgd':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.attack_type == 'fgsm':
        adversary = GradientSignAttack(
            model, loss_fn=criterion, eps=args.test_eps, clip_min=0.0, clip_max=1.0, targeted=False
        )


    start = time.time()
    sum = 0
    sacc = 0
    adv = []
    label = []
    fea = []
    for i, (input, target) in enumerate(val_loader):
        batch = input.size(0)
        input = input.cuda()
        target = target.cuda()

        # adv samples
        if args.attack_type == 'square':
            if args.model_type!='vanilla' and args.model_type!='AT':
                p_init = args.p
                model_sq = ModelPT(model)
                input = input.cpu().numpy()
                target = target.cpu().numpy()
                logits_clean = model_sq.predict(input)
                corr_classified = logits_clean.argmax(1) == target
                y_target_onehot = dense_to_onehot(target, n_cls=args.classes)
                name = 'square/' + args.dataset + '-square-' + args.model_type + '-' + str(args.max_query)
                metrics_path = name + '.metrics'
                log_path = name + '.log'
                n_queries, input_adv, acc = square_attack_linf(model_sq, input, y_target_onehot, corr_classified, args.test_eps,
                                                        n_iters=args.max_query, p_init=p_init, metrics_path=metrics_path, targeted=False,
                                                        loss_type='margin_loss')
                sacc = sacc + acc
                print('{:d}, avg_acc = {:.4f}'.format(i+1, sacc/(i+1)))
                continue
            else:
                input_adv = adversary.run_standard_evaluation(input, target, bs=input.size(0))
            # print(asfsdf)
            # model.set_onlyout(False)
            # _, f = model(input)
            # model.set_onlyout(True)
            # for j in range(input.size(0)):
            #     adv.append(input[j].cpu().detach().numpy())
            #     fea.append(f[j].cpu().detach().numpy())
            # continue
            # visualize_feature_distribution(input_adv, target.cpu().numpy(), model)
            # print(asfsd)
        elif args.attack_type == 'simba':
            input_adv, probs, succs, queries, l2_norms, linf_norms, acc = adversary.simba_batch(
                input, target, args.max_query, args.freq_dims, args.stride, epsilon = 0.2,
                linf_bound=args.test_eps,
                order=args.order, targeted=args.targeted, pixel_attack=args.pixel_attack)
            sacc = sacc + acc
            print('{:d}, avg_acc = {:.4f}'.format(i + 1, sacc / (i + 1)))
            continue
        elif args.attack_type == 'pgd' or args.attack_type == 'fgsm':
            input_adv = adversary.perturb(input, target)

        
        output = model(input_adv)
        loss = criterion(output, target)

        output = output.float()
        loss = loss.float()
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input_adv.size(0))
        top1.update(prec1.item(), input_adv.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                  'Loss: {losses.val:.4f} ({losses.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Time {2:.2f}'.format(
                i, len(val_loader), end - start, losses=losses, top1=top1))
            start = time.time()
        
    # adv = np.array(adv)
    # feature = np.array(fea)
    # print(feature.shape)
    # np.save('adv-sample/' + args.dataset + '/' + args.model_type + '-clean.npy', adv)
    # np.save('adv-sample/' + args.dataset + '/' + args.model_type + '-clean-fea.npy', feature)
    # print(asdsf)

    print('Robust Accuracy {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


