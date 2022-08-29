import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy
from robustbench.utils import load_model
import QueryAttacks.signhunter_original.utils
import torchvision
from advertorch.utils import NormalizeByChannelMeanStd
from mymodel.PreResNet import ResNet18


device = torch.device('cuda:0')

class TargetModel(nn.Module):
    def __init__(self, model, device, args):
        super(TargetModel, self).__init__()
        self.model = model
        self.x_input = torch.zeros((1, 3, 32, 32), device=device)
        self.y_input = torch.tensor(1, dtype=torch.int64, device=device)
        self.pre_softmax = None
        self.device = device
        self.args = args

        self.loss_fn = nn.CrossEntropyLoss(reduce=False).to(device)

    def forward(self, x):
        if self.args.dataset == 'cifar10':
            x = np.transpose(x.astype(np.float32), axes=[0, 3, 1, 2]) / 255.0
        self.x_input = x
        x = torch.tensor(x).cuda()
        self.pre_softmax = self.model(x)
        return self.pre_softmax.cpu().detach().numpy()
    
    def predictions(self, x):
        return np.argmax(self.forward(x), 1)

    def correct_prediction(self, x, y_input):
        if self.args.dataset == 'cifar10':
            return self.predictions(x) == y_input
        elif self.args.dataset == 'imagenet':
            return self.predictions(x) == np.argmax(y_input, 1)
        else:
            print('Correct_prediction False')


    def num_correct(self, x, y_input):
        return np.sum(self.correct_prediction(x, y_input))

    def accuracy(self, x, y_input):
        return np.mean(self.correct_prediction(x, y_input))

    def y_xent(self, x, y_input):
        output = torch.tensor(self.forward(x), device=self.device)
        target = torch.tensor(y_input, device=device)
        return self.loss_fn(output, target).cpu().numpy()

    def xent(self, x, y_input):
        return np.sum(self.loss_fn(self.forward(x), y_input))
    
    def mean_xent(self, x, y_input):
        return np.mean(self.loss_fn(self.forward(x), y_input))


def dense_to_onehot(y_test, n_cls):
    y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
    y_test_onehot[np.arange(len(y_test)), y_test] = True
    return y_test_onehot

def load_cifar10(n_ex, train=False):
    testset = torchvision.datasets.CIFAR10(root='data', train=train, download=True)
    data = np.transpose(testset.data.astype(np.float32), axes=[0, 3, 1, 2]) / 255.0
    label = dense_to_onehot(testset.targets, 10).astype(np.float32)
    return data[:n_ex], label[:n_ex]

def load_net(config, device):
    if config['defense'] is None: model = Model(config['dset_name'], config['model_name'], f'L{config["attack_config"]["p"]}', model_dir='rbmodels', device=device, batch_size=config["inf_batch_size"])
    elif config['defense'] == 'inRND':
        n_in = 0.02 if (config['model_name'] == 'Standard' or config['model_name'] == 'vit_cifar') else 0.05
        model = Model(config['dset_name'], config['model_name'], f'L{config["attack_config"]["p"]}', n_in=n_in, model_dir='rbmodels', device=device, batch_size=config["inf_batch_size"])
    elif config['defense'] == 'outRND':
        n_out = 1 if (config['model_name'] == 'Standard' or config['model_name'] == 'vit_cifar') else 0.3
        model = Model(config['dset_name'], config['model_name'], f'L{config["attack_config"]["p"]}', n_out=n_out, model_dir='rbmodels', device=device, batch_size=config["inf_batch_size"])
    elif config['defense'] == 'AAA':
        if config["aaalr"] > 0:
            model = AAAModel(config['dset_name'], config['model_name'], f'L{config["attack_config"]["p"]}', reverse_step=config["aaalr"], attractor_interval=config["mgiv"], model_dir='rbmodels', device=device, batch_size=config["inf_batch_size"])
        if config["aaalr"] <= 0:
            x_train, y_train = load_cifar10(config["tune_sample_num"], train=True)
            model = AAAModel(config['dset_name'], config['model_name'], f'L{config["attack_config"]["p"]}', attractor_interval=config["mgiv"], model_dir='rbmodels', device=device, batch_size=config["inf_batch_size"])
            model.tune_lr(x_train, y_train)
    elif config['defense'] == 'GS':
        classes = 10
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        model = ResNet18(num_classes=classes, ifgs=False, train_p_epoch=1, lr_p=100)
        model.normalize = dataset_normalization
        model.cuda()
        checkpoint = torch.load('/data1/wyw_Tnnls/weight/PreResNet18/best_val.pth.tar')
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    else: raise NotImplementedError
    target_model = TargetModel(model, device)
    return target_model

def load_net_gs(model, device, args):
    target_model = TargetModel(model, device, args)
    return target_model

def loss(y, logits, targeted=False, loss_type='margin_loss'):
    if loss_type == 'margin_loss':
        preds_correct_class = (logits * y).sum(1, keepdims=True)
        diff = preds_correct_class - logits
        diff[y] = np.inf
        margin = diff.min(1, keepdims=True)
        loss = margin * -1 if targeted else margin
    elif loss_type == 'cross_entropy':
        probs = utils.softmax(logits)
        loss = -np.log(probs[y])
        loss = loss * -1 if not targeted else loss
    else:
        raise ValueError('Wrong loss.')
    return loss.flatten()

def predict(x, model, batch_size, device):
    if isinstance(x, np.ndarray):
        x = np.floor(x * 255.0) / 255.0
        x = x.astype(np.float32)
        n_batches = math.ceil(x.shape[0] / batch_size)
        logits_list = []
        with torch.no_grad():
            for counter in range(n_batches):
                # print('predicting', counter, '/', n_batches, end='\r')
                x_curr = torch.as_tensor(x[counter * batch_size:(counter + 1) * batch_size], device=device)
                logits_list.append(model(x_curr).detach().cpu().numpy())
        logits = np.vstack(logits_list)
        return logits
    else:
        return model(x)

class Model(nn.Module):
    def __init__(self, dataset, arch, norm, model_dir, device=device, batch_size=1000, n_in=0, n_out=0):
        super(Model, self).__init__()
        self.cnn = load_model(model_name=arch, dataset=dataset, threat_model=norm, model_dir=model_dir) if arch != 'vit_cifar' else load_vit_cifar(device)
        self.arch = arch 
        if n_in: self.arch += '_InRND-%.2f' % n_in
        if n_out:self.arch += '_OutRND-%.2f' % n_out
        self.cnn.to(device)
        self.batch_size = batch_size
        self.device = device
        self.loss = loss
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, x):
        noise_in = np.random.normal(scale=self.n_in, size=x.shape)
        logits = predict(np.clip(x + noise_in, 0, 1), self.cnn, self.batch_size, self.device)
        noise_out = np.random.normal(scale=self.n_out, size=logits.shape)
        return logits + noise_out


class AAAModel(nn.Module):
    def __init__(self, dataset, arch, norm, model_dir, device=device, batch_size=1000, attractor_interval=4, dev=0.5, reverse_step=0.06):
        super(AAAModel, self).__init__()
        self.cnn = load_model(model_name=arch, dataset=dataset, threat_model=norm, model_dir=model_dir) if arch != 'vit_cifar' else load_vit_cifar(device)
        self.cnn.to(device)
        self.loss = loss
        self.arch_ori = arch
        self.batch_size = batch_size
        self.device = device
        self.attractor_interval = attractor_interval
        self.reverse_step = reverse_step
        self.dev = dev
        self.arch = 'AAA%s_Lr-%.2f-Ai-%d' % (self.arch_ori, self.reverse_step, self.attractor_interval)

        def hook(module, fea_in, fea_out): self.feature = fea_in[0]
        fc_names = []
        for name, module in self.cnn.named_modules():
            if 'fc' in name or 'logits' in name: fc_names.append(name)
        #print(fc_names)
        for name, module in self.cnn.named_modules():
            if name == fc_names[-1]: self.hook = module.register_forward_hook(hook=hook)

    def set_hp(self, reverse_step, attractor_interval=4):
        self.attractor_interval = attractor_interval
        self.reverse_step = reverse_step
        self.arch = 'AAA%s_Lr-%.2f-Ai-%d' % (self.arch_ori, self.reverse_step, self.attractor_interval)
    
    def tune_lr(self, x_train, y_train):
        alpha = (0.01 / 255) if self.arch_ori == 'Standard' else (0.1 / 255)
        reverse_step_interval = 0.01
        attractor_interval_interval = 1
        stop_threshold_ascend = 0.9
        acc_drop_tolerance = 0.001

        y_train_np = deepcopy(y_train[:self.batch_size].astype(bool))
        advs = [deepcopy(x_train)]
        x_train = torch.as_tensor(x_train[:self.batch_size], device=self.device)
        y_train = torch.as_tensor(y_train[:self.batch_size].astype(bool), device=self.device)
        
        margin = np.array(1)
        x_train.requires_grad = True
        with torch.enable_grad():
            while margin.mean() > 0:
                self.cnn.zero_grad()
                logits = self.cnn(x_train)
                gt_out = (logits * y_train).max(1)[0]
                sc_out = (logits * ~y_train).max(1)[0]
                margin = gt_out - sc_out
                if len(advs) == 1: acc_ori = (margin.detach().cpu().numpy() > 0).mean()
                grad = torch.autograd.grad(margin.mean(), x_train)[0]
                x_train = torch.clamp(x_train - grad.sign() * alpha, 0, 1)
                x_train.grad = None
                advs.append(x_train.detach().cpu().numpy())
                print('crafting tuning samples', 'margin [%.2f, %.2f, %.2f]' % (margin.min(), margin.mean(), margin.max()), end='\r')

        print()
        attractor_interval = self.attractor_interval
        ascend_ratio = 0
        acc = 0
        while acc_ori - acc > acc_drop_tolerance:
            reverse_step = 0
            acc = acc_ori
            while (ascend_ratio < stop_threshold_ascend) and (acc_ori - acc <= acc_drop_tolerance):
                reverse_step += reverse_step_interval
                self.set_hp(reverse_step, attractor_interval)
                loss = None
                is_ascend = []
                for i in range(len(advs)):
                    logits = self.forward(advs[i], verbose=False)
                    gt_out = (logits * y_train_np).max(1)
                    sc_out = (logits * ~y_train_np).max(1)
                    margin = gt_out - sc_out
                    if loss is None: acc = (margin > 0).mean()
                    else: is_ascend.append(margin > loss)
                    loss = deepcopy(margin)
                    #print('predicting', i, '/', len(advs), end='\r')
                ascend_ratio = np.concatenate(is_ascend, axis=0).mean()
                print('mgiv %d' % attractor_interval, 'lr %.2f' % reverse_step, 'ascend %.2f' % (ascend_ratio * 100))
            attractor_interval += attractor_interval_interval
        return self.reverse_step, self.attractor_interval

    def forward(self, x, verbose=False):
        if isinstance(x, np.ndarray):
            if verbose: x = np.floor(x * 255.0) / 255.0
            x = x.astype(np.float32)
        else:
            x.requires_grad = True
        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []

        try: self.cnn.fc
        except AttributeError: self.cnn.fc = self.cnn.logits
        if isinstance(x, np.ndarray):
            for counter in range(n_batches):
                with torch.no_grad():
                    if verbose: print('predicting', counter, '/', n_batches, end='\r')
                    x_curr = torch.as_tensor(x[counter * self.batch_size:(counter + 1) * self.batch_size], device=self.device)
                    self.cnn(x_curr) # get feature by hook
                    feature = self.feature
                with torch.enable_grad():
                    feature.requires_grad = True
                    prob = self.cnn.fc(feature) 
                    value, index = torch.topk(prob, k=2, dim=1)

                    margin = value[:, 0] - value[:, 1]
                    target = ((margin / self.attractor_interval + self.dev).round() - self.dev) * self.attractor_interval
                    grad = torch.autograd.grad(margin.sum(), feature, create_graph=True, retain_graph=True)[0]
                    
                    feature = feature + (target - margin)[:, None] * grad * self.reverse_step
                    logits = self.cnn.fc(feature)
                    logits_list.append(logits.detach().cpu())
            logits = torch.vstack(logits_list)
            return logits.numpy()
        else:
            with torch.enable_grad():
                x.requires_grad = True
                self.cnn(x)
                feature = self.feature
                prob = torch.nn.functional.softmax(self.cnn.fc(feature), 1)
                value, index = torch.topk(prob, k=2, dim=1)
                
                margin = value[:, 0] - value[:, 1]
                target = ((margin / self.attractor_interval + self.dev).round() - self.dev) * self.attractor_interval
                grad = torch.autograd.grad(margin.sum(), feature, create_graph=True, retain_graph=True)[0]
                
                feature = feature + (target - margin)[:, None] * grad * self.reverse_step
                self.cnn.zero_grad()
                logits = self.cnn.fc(feature)
                return logits


def load_vit_cifar(device):
    cnn = ViT(device)
    cnn.load_state_dict(state_dict=torch.load("data/epoch=53-step=5291.tmp_end.ckpt"))
    cnn.fc = cnn.model.fc
    return cnn

import pytorch_lightning as pl
class ViT(pl.LightningModule):
    def __init__(self, device):
        super(ViT, self).__init__()
        self.model = vit(mlp_hidden=384, head=12)
        self.mean, self.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        self.mean, self.std = torch.as_tensor(self.mean, device=device), torch.as_tensor(self.std, device=device)

    def forward(self, x):
        return self.model((x - self.mean[None, :, None, None]) / self.std[None, :, None, None])

class vit(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0., num_layers:int=7, hidden:int=384, mlp_hidden:int=384*4, head:int=8, is_cls_token:bool=True):
        super(vit, self).__init__()
        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))
        enc_list = [TransformerEncoder(hidden,mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token: out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token: out = out[:,0]
        else: out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o