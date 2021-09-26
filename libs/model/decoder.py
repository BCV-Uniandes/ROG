import torch.nn as nn
import torch.nn.functional as F

from libs.model.encoder import Block, SeparableConv3d, Norm_layer


class Conv3D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, padding=1, stride=1,
                 norm_layer=None):
        super(Conv3D, self).__init__()
        self.conv = SeparableConv3d(
            inplanes, planes, kernel_size, padding, 1, 1)
        self.norm = Norm_layer(norm_layer, planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(self.relu(x))
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, filters, reps, strides, norm_layer):
        super(Decoder, self).__init__()
        self.normalization = norm_layer
        self.strides = strides
        blocks = []
        combination = []
        # breakpoint()
        for i in reversed(range(2, len(filters))):
            combination.append(Conv3D(
                filters[i], filters[i - 1], 3, 1, 1, norm_layer))
            blocks.append(Block(
                filters[i - 1], filters[i - 1], reps[i - 1], 1, 1, norm_layer))

        combination.append(Conv3D(
            filters[1], filters[0], 3, 1, 1, norm_layer))
        self.combination = nn.ModuleList(combination)
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Conv3d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=False)
        self._init_weights()

    def forward(self, x):
        x_ = x.pop()
        num = len(self.strides) - 1
        for idx, (i, j) in enumerate(zip(self.blocks, self.combination)):
            x_ = j(x_)
            x_ = F.interpolate(
                input=x_, scale_factor=self.strides[num - idx],
                mode='trilinear', align_corners=True)
            x_ += x.pop()
            x_ = i(x_)
        x_ = self.combination[-1](x_)
        x_ = F.interpolate(
            input=x_, scale_factor=self.strides[0],
            mode='trilinear', align_corners=True)
        x = self.final(x_)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, a=1e-2)
                if m.bias is not None:
                    m.bias = nn.init.constant_(m.bias, 0)
            elif isinstance(m, self.normalization):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
