import torch.nn as nn


class Norm_layer(nn.Module):
    def __init__(self, normalization, planes):
        super(Norm_layer, self).__init__()
        self.flag = False
        self.bn = normalization(planes, affine=True)
        self.aux_bn = normalization(planes, affine=True)

    def forward(self, x):
        if self.flag:
            x = self.bn(x)
        else:
            x = self.aux_bn(x)
        return x

    def set_flag(self, flag):
        self.flag = flag


class SeparableConv3d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, padding=1, stride=1,
                 dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, inplanes, kernel_size, stride,
                               padding, dilation, groups=inplanes, bias=bias)
        self.pointwise = nn.Conv3d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride, dilation, norm_layer):
        super(Block, self).__init__()
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv3d(inplanes, planes, 1, stride=stride,
                                  bias=False)
            self.skip_norm = Norm_layer(norm_layer, planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        layers = [self.relu]

        if stride != 1:
            layers.append(SeparableConv3d(
                inplanes, planes, 3, 1, stride, dilation))
            layers.append(Norm_layer(norm_layer, planes))
        else:
            layers.append(SeparableConv3d(
                inplanes, planes, 3, 1, 1, dilation))
            layers.append(Norm_layer(norm_layer, planes))

        for _ in range(reps - 1):
            layers.append(self.relu)
            layers.append(SeparableConv3d(
                planes, planes, 3, 1, 1, dilation))
            layers.append(Norm_layer(norm_layer, planes))

        self.layers = nn.Sequential(*layers)

    def forward(self, inp):
        x = self.layers(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skip_norm(skip)
        else:
            skip = inp
        x = x + skip
        return x


class Backbone(nn.Module):
    """
    Based on the modified Alighed Xception
    """
    def __init__(self, modalities, filters, strides, repetitions,
                 dilations, norm_layer):
        super(Backbone, self).__init__()
        self.normalization = norm_layer
        self.conv1 = nn.Conv3d(modalities, filters[0], 3, padding=1,
                               bias=False)
        self.bn1 = Norm_layer(norm_layer, filters[0])
        self.relu = nn.ReLU(inplace=True)

        # Is this one necessary?
        self.conv2 = nn.Conv3d(filters[0], filters[0], 3, padding=1,
                               bias=False)
        self.bn2 = Norm_layer(norm_layer, filters[0])

        blocks = []
        for i in range(len(filters) - 1):
            blocks.append(Block(filters[i], filters[i + 1], repetitions[i],
                                strides[i], dilations[i], norm_layer))
        self.blocks = nn.ModuleList(blocks)
        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = [x]
        for i in self.blocks:
            x.append(i(x[-1]))
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
