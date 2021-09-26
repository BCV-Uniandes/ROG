import torch.nn as nn
from libs.model.decoder import Decoder
from libs.model.encoder import Backbone


class GNet(nn.Module):
    def __init__(self, params):
        super(GNet, self).__init__()
        self.normalization = params['normalization']
        self.backbone = Backbone(
            params['modalities'], params['filters'], params['strides'],
            params['repetitions'], params['dilations'], self.normalization)
        self.decoder = Decoder(
            params['classes'], params['filters'], params['repetitions'],
            params['strides'], self.normalization)

        print('A GNet with the following characteristics was initialized:\n'
              ' - Input modalities: {}\n - classes: {}\n - filters: {}\n'
              ' - strides: {}\n - repetitions: {}\n - dilations: {}'.format(
                  params['modalities'], params['classes'],
                  params['filters'], params['strides'],
                  params['repetitions'], params['dilations']))

        print('Number of parameters: {}'.format(
            sum([p.data.nelement() for p in self.backbone.parameters()]) +
            sum([p.data.nelement() for p in self.decoder.parameters()])))

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x
