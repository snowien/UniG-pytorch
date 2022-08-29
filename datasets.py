import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST
from advertorch.utils import NormalizeByChannelMeanStd
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

__all__ = ['cifar10_dataloaders', 'sub_cifar10_dataloader', 'imagenet_dataloader']


# prepare imagenet
def move_valimg(val_dir='./data/ImageNet2012/ILSVRC2012_img_val', devkit_dir='./data/ImageNet2012/ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12'):
    """
     move valimg to correspongding folders.
     val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
     organize like:
     /val
        /n01440764
            images
        /n01443537
            images
         .....
     """
    ## load synset, val ground truth and val images list
    synset = scipy.io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))

    ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]

    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id - 1]
        WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

        # move val images
        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))


def select_imagenet_1000(n_ex=1000, model=None, seed=0, type='None'):
    with open('data/val.txt', 'r') as f:
        txt = f.read().split('\n')
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
        input = torch.tensor(img[np.newaxis, ...]).cuda()
        prd = model(input).cpu().numpy().argmax(1)
        if prd != lbl: continue

        label[len(data), lbl] = 1
        data.append(img)
        label_done.append(lbl)
        print('selecting samples in different classes...', len(label_done), '/', 1000, end='\r')
        if len(label_done) == min([1000, n_ex]): break

    x_test = np.array(data)
    y_test = np.array(label)
    np.save('data/imagenet_wide_resnet50_2_' + type + '_imgs_0.npy', x_test)
    np.save('data/imagenet_wide_resnet50_2_' + type + '_lbls_0.npy', y_test)
    return


def select_cifar10_100(test_loader, model, num=100, type='None'):
    model.eval()
    data = []
    label = [] 
    for i, (input, target) in enumerate(test_loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct.cpu().squeeze().numpy()
        index = np.argwhere(correct==True).flatten()
        for j in index:
            if len(data)<num:
                label.append(target[j].cpu().numpy())
                data.append(input[j].cpu().numpy())
            else:
                x_test = np.array(data)
                y_test = np.array(label)
                print(x_test.shape, y_test.shape)
                np.save('data/cifar10_pre_resnet18_' + type + '_imgs_0.npy', x_test)
                np.save('data/cifar10_pre_resnet18_' + type + '_lbls_0.npy', y_test)
                return


def cifar10_dataloader(args):
    args.classes = 10
    test_set = CIFAR10('data/cifar10', train=False, transform=transforms.Compose([transforms.ToTensor(),]), download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_set = Subset(CIFAR10('data/cifar10', train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10('data/cifar10', train=True, transform=transforms.Compose([transforms.ToTensor(),]), download=True), list(range(45000, 50000)))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return test_loader


class MyDataset(Dataset):
    def __init__(self, data, label, name):
        self.data = np.load(data)
        self.data = torch.tensor(self.data)
        self.label = np.load(label)
        self.label = torch.tensor(self.label)
        if name == 'imagenet':
            self.label = torch.argmax(self.label, -1)

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.label[index]

    def __len__(self):
        return self.data.shape[0]
    
    
def sub_cifar10_dataloader(args):
    args.classes = 10
    if args.model_type == 'AT':
        subtestset = MyDataset('data/cifar10_pre_resnet18_'+args.model_type+'_imgs_0.npy',
                            'data/cifar10_pre_resnet18_'+args.model_type+'_lbls_0.npy', args.dataset)
    elif args.model_type == 'vanilla' or args.model_type == 'GS' or args.model_type == 'DENT':
        subtestset = MyDataset('data/cifar10_pre_resnet18_' + 'vanilla1000' + '_imgs_0.npy',
                                'data/cifar10_pre_resnet18_'+ 'vanilla1000' +'_lbls_0.npy', args.dataset)
    test_loader = torch.utils.data.DataLoader(subtestset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                              pin_memory=False, generator=torch.Generator(device='cuda'))
    return test_loader


def imagenet_dataloader(args):
    args.classes = 1000
    imgset = datasets.ImageFolder('./data/ImageNet2012/ILSVRC2012_img_val', transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))
    # np.random.seed(10)
    # split_permutation = list(np.random.permutation(50000))
    # subimgset = Subset(imgset, split_permutation[:1000])
    test_loader = torch.utils.data.DataLoader(
        imgset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False)
    if args.model_type == 'vanilla':
        subimgset = MyDataset('./data/imagenet_wide_resnet50_2_imgs_0.npy',
                                'data/imagenet_wide_resnet50_2_lbls_0.npy', args.dataset)
    elif args.model_type == 'AT':
        subimgset = MyDataset('./data/imagenet_wide_resnet50_2_AT_imgs_0.npy',
                                'data/imagenet_wide_resnet50_2_AT_lbls_0.npy', args.dataset)
    elif args.model_type == 'RND':
        subimgset = MyDataset('./data/imagenet_wide_resnet50_2_RND_imgs_0.npy',
                                'data/imagenet_wide_resnet50_2_RND_lbls_0.npy', args.dataset)
    elif args.model_type == 'UniG':
        subimgset = MyDataset('./data/imagenet_wide_resnet50_2_imgs_0.npy',
                                'data/imagenet_wide_resnet50_2_lbls_0.npy', args.dataset)
    elif args.model_type == 'UniGAT':
        subimgset = MyDataset('./data/imagenet_wide_resnet50_2_AT_imgs_0.npy',
                                'data/imagenet_wide_resnet50_2_AT_lbls_0.npy', args.dataset)
    elif args.model_type == 'DENT':
        subimgset = MyDataset('./data/imagenet_wide_resnet50_2_imgs_0.npy',
                                'data/imagenet_wide_resnet50_2_lbls_0.npy', args.dataset)
    subtest_loader = torch.utils.data.DataLoader(
        subimgset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False)
    return test_loader, subtest_loader


if __name__ == '__main__':
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
        
    select_cifar10_100(test_loader, model, num=1000, type=args.model_type+'1000')
    select_imagenet_1000(model=model, type=args.model_type)
    

'''
def mnist_dataloaders(batch_size=64, data_dir = 'datasets/mnist'):
    transform = transforms.Compose(
        [transforms.ToTensor(), ])

    train_set = Subset(MNIST(data_dir, train=True, transform=transform, download=True), list(range(45000)))
    val_set = Subset(MNIST(data_dir, train=True, transform=transform, download=True), list(range(45000, 50000)))
    test_set = MNIST(data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def cifar10_dataloaders(batch_size=64, data_dir = 'datasets/cifar10'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def cifar100_dataloaders(batch_size=64, data_dir = 'datasets/cifar100'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def tiny_imagenet_dataloaders(batch_size=64, data_dir = 'datasets/tiny-imagenet-200', permutation_seed=10):

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'training')
    val_path = os.path.join(data_dir, 'validation')

    np.random.seed(permutation_seed)
    split_permutation = list(np.random.permutation(100000))

    train_set = Subset(ImageFolder(train_path, transform=train_transform), split_permutation[:90000])
    val_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[90000:])
    test_set = ImageFolder(val_path, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
'''