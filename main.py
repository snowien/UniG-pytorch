# -*- coding: utf-8 -*-
# @Time    : 2022.08.28
# @Author  : Yingwen Wu
# @File    : main.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import argparse
import torch.optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
torch.set_default_tensor_type('torch.cuda.FloatTensor')

### import datasets
from datasets import cifar10_dataloader,sub_cifar10_dataloader,imagenet_dataloader
from utils import *
#### import models
from model.PreResNet import ResNet18
from model.dent import DENTModel
from model.UniG import UniGModel
from model.RND import RNDModel
from model.CVPR_2019_PNI.code.models.noisy_resnet_cifar import noise_resnet20
# import torchvision.models as models
from model.Wide_ResNet import wide_resnet50_2
# from robustbench.utils import load_model
from model.PreResNet_RobBen import DMPreActResNet, Swish


class Logger(object):
    def __init__(self, logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def set_args():
    parser = argparse.ArgumentParser(description='PyTorch Standard Training')

    ########################## data setting ##########################
    parser.add_argument('--data', type=str, default='data/ImageNet', help='location of the data corpus')#, required=True)
    parser.add_argument('--dataset', type=str, default='ImageNet', help='dataset [cifar10, cifar100, tinyimagenet]')#, required=True)

    ########################## model setting ##########################
    parser.add_argument('--arch', type=str, default='PreResNet18', help='model architecture [resnet18, wideresnet, vgg16]')#, required=True)
    parser.add_argument('--model_type', type=str, default='vanilla', help='UniG,AT,RND,GSAT,PNI')  # , required=True)
    parser.add_argument('--classes', default=10, type=int, help='data classes')
    parser.add_argument('--depth_factor', default=34, type=int, help='depth-factor of wideresnet')
    parser.add_argument('--width_factor', default=10, type=int, help='width-factor of wideresnet')

    ########################## basic setting ##########################
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='gpu device id')
    parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained model')
    parser.add_argument('--eval', action="store_true", help="evaluation pretrained model")
    parser.add_argument('--print_freq', default=50, type=int, help='logging frequency during training')
    parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)

    ########################## training setting ##########################
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--decreasing_lr', default='50,100,150', help='decreasing strategy')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--local_rank', default = -1, type = int, help ='node rank for distributed training')
    # UniG
    parser.add_argument('--lr_', default=0.1, type=float, help='gs initial learning rate')
    parser.add_argument('--epochs_', default=10, type=int, help='gs number of total epochs to run')
    parser.add_argument('--delta', default=0.2, type=float, help='gs parameter delta')
    parser.add_argument('--combine', action="store_true", help="UniG batch size=1 with combine")
    # RND
    parser.add_argument('--n_in', default=0.02, type=float, help='RND noise')
    # Gaussian
    parser.add_argument('--std', default=0.5, type=float, help='Gau Model feature multiple noise')

    ########################## attack setting ##########################
    parser.add_argument('--attack_type', default='square', type=str, help='attack type')
    parser.add_argument('--norm', default='Linf', type=str, help='Linf or l2 or fgsm')
    parser.add_argument('--test_eps', default=8/255, type=float, help='epsilon of attack during testing')
    parser.add_argument('--max_query', default=2500, type=int, help='max query time')
    parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
    parser.add_argument('--test_gamma', default=2/255, type=float, help='step size of attack during testing')
    parser.add_argument('--test_randinit', action='store_false', help='randinit usage flag (default: on)')
    # square
    parser.add_argument('--p', default=0.05, type=float, help='square attack setting')
    # simba
    parser.add_argument('--freq_dims', type=int, default=32, help='dimensionality of 2D frequency space')
    parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
    parser.add_argument('--stride', type=int, default=7, help='stride for block order')
    parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
    parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
    # bandits
    parser.add_argument('--fd-eta', type=float, help='\eta, used to estimate the derivative via finite differences')
    parser.add_argument('--image-lr', type=float, help='Learning rate for the image (iterative attack)')
    parser.add_argument('--online-lr', type=float, help='Learning rate for the prior')
    parser.add_argument('--mode', type=str, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--exploration', type=float, help='\delta, parameterizes the exploration to be done around the prior')
    parser.add_argument('--tile-size', type=int, help='the side length of each tile (for the tiling prior)')
    parser.add_argument('--json_config', type=str, default='QueryAttacks-json/cifar10-bandits-linf.json', help='a config file to be passed in instead of arguments')
    parser.add_argument('--log-progress', default=True)
    parser.add_argument('--nes', action='store_true')
    parser.add_argument('--tiling', action='store_true')
    parser.add_argument('--gradient-iters', type=int)
    parser.add_argument('--total-images', type=int)

    args = parser.parse_args()
    return args


def main():
    args = set_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.model_type != 'PNI':
        torch.multiprocessing.set_start_method('spawn')
        
    if args.model_type=='UniG':
        sys.stdout = Logger('./log/' + args.model_type + '-' + args.dataset + '-query' + str(args.max_query)
                        + '-bs' + str(args.batch_size) + '-delta' + str(args.delta) + '-ep' + str(args.epochs_) 
                        + '-lr' + str(args.lr_) + '.log')
    elif args.model_type=='RND':
        sys.stdout = Logger('./log/' + args.model_type + '-' + args.dataset + '-query' + str(args.max_query) + '-std' + str(args.n_in) + '.log')
    else:
        sys.stdout = Logger('./log/' + args.model_type + '-' + args.dataset + '-query' + str(args.max_query) + '.log')
    print(args)
    
    if args.seed:
        print('set random seed = ', args.seed)
        setup_seed(args.seed)

    ### load data: cifar10, subcifar10, imagenet
    if args.dataset == 'cifar10':
        test_loader = cifar10_dataloader(args)
    elif args.dataset == 'subcifar10':
        test_loader = sub_cifar10_dataloader(args)
        args.dataset = 'cifar10'
    elif args.dataset == 'imagenet':
        test_loader, subtest_loader = imagenet_dataloader(args)
    else:
        print('False dataset!')

    #### load model: GS,AT,RND,Gas,vanilla
    if args.model_type == 'vanilla':
        if args.dataset == 'cifar10':
            args.pretrained = 'weight/PreResNet18/best_val.pth.tar'
            model = load_cifar10_model(args.pretrained)
            model.set_onlyout(True)
        else:
            model = wide_resnet50_2(pretrained=True)
            model.set_onlyout(True)
    elif args.model_type == 'AT':
        if args.dataset == 'cifar10':
            args.arch = 'Gowal2021Improving_R18_ddpm_100m'
            checkpoint = torch.load('./weight/rbmodels/cifar10/Linf/' + args.arch + '.pt')
            model = DMPreActResNet(num_classes=10, depth=18, width=0, activation_fn=Swish, mean= (0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
            model.load_state_dict(checkpoint, strict=True)
            # model = load_model(model_name=args.arch, dataset='cifar10', threat_model='Linf', model_dir='./rbmodels')
            model.set_onlyout(True)
        else:
            model = wide_resnet50_2()
            model.set_onlyout(True)
            state_dict = torch.load('./weight/rbmodels/imagenet/Linf/Salman2020Do_50_2.pt')
            model.load_state_dict(state_dict, strict=False)
    elif args.model_type == 'UniG':
        if args.dataset == 'cifar10':
            args.pretrained = 'weight/PreResNet18/best_val.pth.tar'
            net = load_cifar10_model(args.pretrained)
            model = UniGModel(net, epoch=args.epochs_, lr=args.lr_, shape=(args.batch_size, 512) if not args.combine else (args.batch_size+5, 512), delta=args.delta, ifcombine=args.combine)
        else:
            model = UniGModel(wide_resnet50_2(pretrained=True), epoch=args.epochs_, lr=args.lr_, shape=(args.batch_size, 2048) if not args.combine else (args.batch_size+5, 2048), delta=args.delta, ifcombine=args.combine)
    elif args.model_type == 'UniGAT':
        if args.dataset == 'cifar10':
            args.arch = 'Gowal2021Improving_R18_ddpm_100m'
            checkpoint = torch.load('./weight/rbmodels/cifar10/Linf/' + args.arch + '.pt')
            net = DMPreActResNet(num_classes=10, depth=18, width=0, activation_fn=Swish, mean= (0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
            net.load_state_dict(checkpoint, strict=True)
            net.set_onlyout(True)
            # net = load_model(model_name=args.arch, dataset='cifar10', threat_model='Linf', model_dir='./rbmodels')
            model = UniGModel(net, epoch=args.epochs_, lr=args.lr_, shape=(args.batch_size, 512) if not args.combine else (args.batch_size+5, 512), delta=args.delta, ifcombine=args.combine)
        else:
            net = wide_resnet50_2()
            state_dict = torch.load('./weight/rbmodels/imagenet/Linf/Salman2020Do_50_2.pt')
            net.load_state_dict(state_dict, strict=False)
            model = UniGModel(net, epoch=args.epochs_, lr=args.lr_,
                            shape=(args.batch_size, 2048), delta=args.delta)
    elif args.model_type == 'RND':
        if args.dataset == 'cifar10':
            args.pretrained = 'weight/PreResNet18/best_val.pth.tar'
            net = load_cifar10_model(args.pretrained)
            net.set_onlyout(True)
            model = RNDModel(net, std=args.n_in)
            print('RND-n_in: ', args.n_in)
        else:
            net = wide_resnet50_2(pretrained=True)
            net.set_onlyout(True)
            model = RNDModel(net, std=args.n_in)
            print('RND-n_in: ', args.n_in)
    elif args.model_type == 'DENT':
        if args.dataset == 'cifar10':
            args.pretrained = 'weight/PreResNet18/best_val.pth.tar'
            net = load_cifar10_model(args.pretrained)
            net.set_onlyout(True)
            model = DENTModel(net)
            print('Load Dent cifar10')
            # model.set_onlyout(True)
        else:
            net = wide_resnet50_2(pretrained=True)
            net.set_onlyout(True)
            model = DENTModel(net)
            print('Load Dent imagenet')
    elif args.model_type == 'PNI':
        if args.dataset == 'cifar10':
            net = noise_resnet20()
            model = torch.nn.Sequential(
                Normalize_layer(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
                net
            )
            checkpoint = torch.load('weight/PNI/checkpoint.pth.tar')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('Load PNI cifar10')
        else:
            print('can not conduct PNI on imagenet')
            return
    elif args.model_type == 'Gas':
        if args.dataset == 'cifar10':
            args.pretrained = 'weight/PreResNet18/best_val.pth.tar'
            net = load_cifar10_model(args.pretrained)
            model = GauModel(net, std=args.std)
        else:
            net = wide_resnet50_2(pretrained=True)
            model = GauModel(net, std=args.std)

    
    if torch.cuda.device_count() > 1:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    else:
        model.cuda()


    criterion = nn.CrossEntropyLoss()
    # plot_margin_loss()
    # plot_output_diff()
    ######################### only evaluation ###################################
    if args.eval:
        model.eval()
        print('------ test ' + args.model_type + ' model, batch=' + str(args.batch_size) + '------')
        with torch.no_grad():
            # select_cifar10_100(test_loader, model, num=1000, type=args.model_type)  
            # select_imagenet_1000(model=model, type=args.model_type)
            # grad = get_input_gradient(test_loader, model, args.dataset + '-' + args.model_type)
            # logits = get_output(test_loader, model, args.dataset + '-' + args.model_type)
            # cal_logit_diff('./model_output/' + args.dataset + '-vanilla.npy', './model_output/' + args.dataset + '-' + args.model_type + '.npy')
            # plot_hyper_param(batch, c_batch, d_batch, r_batch, 'batch')
            # plot_hyper_param(delta, c_delta, d_delta, r_delta, 'delta')
            # plot_hyper_param(lr, c_lr, d_lr, r_lr, 'lr')
            # plot_hyper_param(epoch, c_epoch, d_epoch, r_epoch, 'epoch')
           
            acc = test(test_loader, model, criterion, args)
            print('clean acc:', acc)
            attacks = ['square', 'simba', 'signhunter', 'nes', 'bandits']
            print(attacks)
            for a in attacks:
                if args.dataset == 'cifar10':
                    if args.norm == 'Linf':
                        args.test_eps = 8/255
                    if args.norm == 'L2' and a =='square':
                        args.test_eps = 1
                    subtest_loader = test_loader
                elif args.dataset == 'imagenet':
                    if args.norm == 'Linf':
                        args.test_eps = 4/255
                    if args.norm == 'L2' and a == 'square':
                        args.test_eps = 5
                args.attack_type = a
                print('------' + args.attack_type + '-' + args.norm + '------')
                adv_acc = test_adv(subtest_loader, model, criterion, args)
            return



if __name__ == '__main__':
    main()
    ## batch, c_batch, d_batch, r_batch, 'batch'
    # batch = [32, 64, 128, 256, 512]
    # c_batch = [94.26, 94.26, 94.25, 94.26, 94.26]
    # d_batch = [1.4416, 1.0788, 0.9454, 0.8544, 0.8053]
    # r_batch = [78.71, 80.47, 80.95, 80.00, 80.22]

    # ## delta, c_delta, d_delta, r_delta, 'delta'
    # delta = [0.1, 0.3, 0.5, 0.7, 1]
    # c_delta = [94.26, 94.26, 94.26, 94.26, 94.32]
    # d_delta = [0.199, 0.554, 0.818, 1.017, 1.290]
    # r_delta = [70.70, 75.41, 80.12, 82.95, 80.12]

    # ### lr, c_lr, d_lr, r_lr, 'lr'
    # lr = [1, 5, 10, 50, 100]
    # c_lr = [94.26, 94.26, 94.26, 94.26, 94.26]
    # d_lr = [0.652, 0.783, 0.845, 0.902, 1.418]
    # r_lr = [80.12, 82.01, 80.12, 80.12, 80.12]

    # ### epoch, c_epoch, d_epoch, r_epoch, 'epoch'
    # epoch = [1, 3, 5, 7, 10]
    # c_epoch = [94.26, 94.26, 94.26, 94.26, 94.26]
    # d_epoch = [0.818, 0.850, 0.913, 1.010, 1.141]
    # r_epoch = [80.12, 79.18, 77.29, 77.29, 76.35]

