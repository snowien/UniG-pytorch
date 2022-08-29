class GauModel(nn.Module):
    def __init__(self, model, std=0.5):
        super(GauModel, self).__init__()
        self.model = model
        self.std = std

    def forward(self, x):
        _, fea = self.model(x)
        fea_noise = self.multi_gaussian_noise_to_fea(fea)
        out = self.model.linear_layer(fea_noise)
        return out

    def multi_gaussian_noise_to_fea(self, x):
        noise = x.clone().normal_(1, self.std)
        print('max:', noise.max())
        print('min:', noise.min())
        out = x*noise
        return out

    def forward_undefend(self, x):
        out, fea = self.model(x)
        return out