import torch
import torch.optim
import torch.nn as nn
import numpy as np

class UniGModel(nn.Module):
    def __init__(self, model, shape, epoch, lr, delta, ifcombine):
        super(UniGModel, self).__init__()
        self.model = model
        self.epoch = epoch
        self.lr = lr
        self.delta = delta
        self.ifcombine = ifcombine

        self.model.set_gs_param(shape)
        self.optimizer = torch.optim.SGD([self.model.gs.p],
                                         lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

        for name, param in self.model.named_parameters():
            if 'gs' not in name:
                param.requires_grad = False
        # # train data
        train_input = np.load('data/cifar10_pre_resnet18_GS_train_imgs_0.npy')
        self.train_input = torch.from_numpy(train_input).cuda()
        # # train label
        train_label = np.load('data/cifar10_pre_resnet18_GS_train_lbls_0.npy')
        self.train_label = torch.from_numpy(train_label).cuda()
        self.addition = 5

    def get_target(self, output):
        _, pred = output.topk(1, 1, True, True)
        target = pred.squeeze().detach()
        return target
    
    def combine(self, x):
        index = int(100 * torch.rand(size=[1]))
        input = torch.cat((x, self.train_input[index:index + self.addition]), 0)
        train_label = self.train_label[index:index+self.addition]
        return input, train_label

    def gs_loss(self, grad):
        grad_simi = 0
        for j in range(grad.size(0) - 1):
            grad_simi += torch.norm((grad[j] - grad[j + 1]))
        return grad_simi

    def forward(self, x):
        batch = x.size(0)
        if self.ifcombine:
            x, train_label = self.combine(x)
        self.model.gs.init_param()
        x.requires_grad_()
        with torch.enable_grad():
            with torch.no_grad():
                fea = self.model.get_feature(x)
                output2 = self.model.undefend_get_pred(fea)
                target = self.get_target(output2)
                if self.ifcombine:
                    target = torch.cat((target[0:batch], train_label), 0)
            for i in range(self.epoch):
                #### forward
                # fea = self.model.get_feature(x)
                fea.requires_grad_()
                output = self.model.get_pred(fea)
                # output, fea, _ = self.model(x)
                ### optimize
                #----- gs loss
                loss_ce = self.criterion(output, target)
                self.optimizer.zero_grad()
                grad = torch.autograd.grad(loss_ce, fea, create_graph=True, retain_graph=True)[0]
                if i == 0:
                    min = grad.min().detach()
                    max = grad.max().detach()
                grad = (grad - min) / (max - min)
                loss_gs = self.gs_loss(grad)
                #----- all loss
                loss = 1 * loss_gs
                #----- step
                self.optimizer.zero_grad()
                # print(loss)
                loss.backward()
                self.optimizer.step()
                self.model.gs.p.data = torch.clamp(self.model.gs.p.data, 1 - self.delta, 1 + self.delta)
                # print('epoch:{:d}, gs_loss: {:.3f}, fea_loss: {:.3f}'.format(i, loss_gs.data, loss_fea.data))
        with torch.no_grad():
            output = self.model.get_pred(fea)     
        return output[0:batch]
       
    def forward_undefend(self, x):
        out, fea, _ = self.model.forward_undefend(x)
        return out

    def softmax(self, x, axis=1):
        x_, _ = torch.max(x, axis, keepdim=True)
        x = x - x_
        y = torch.exp(x)
        return y / torch.sum(y, axis, keepdim=True)
    