import torch
import torch.nn as nn

class RNDModel(nn.Module):
    def __init__(self, model, std=0.02):
        super(RNDModel, self).__init__()
        self.model = model
        self.std = std

    def forward(self, x):
        x = self.inrnd(x)  ### RND method
        output = self.model.forward_undefend(x)
        return output

    def inrnd(self, x):
        noise_in = x.detach().clone().normal_(0, self.std)
        x = torch.clip(x + noise_in, 0, 1)
        return x

    def forward_undefend(self, x):
        out = self.model(x)
        return out
