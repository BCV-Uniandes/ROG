import torch
import torch.nn as nn
import torch.nn.functional as F


class swish(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)


class Norm_layer(nn.Module):
    def __init__(self, planes):
        super(Norm_layer, self).__init__()
        self.norm = nn.InstanceNorm3d(planes, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x


class SeparableConv3d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, padding=1, stride=1,
                 dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()
        self.conv = nn.Conv3d(
            inplanes, inplanes, kernel_size, stride, padding, dilation,
            groups=inplanes, bias=bias)
        self.pointwise = nn.Conv3d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x


class Cell(nn.Module):
    def __init__(self, inplanes, up=True, same=True, down=True, factor=2):
        super(Cell, self).__init__()
        self.factor = factor
        self.up, self.down = None, None
        self.same = same
        if up:
            self.up = nn.Sequential(
                swish(),
                nn.Conv3d(inplanes, inplanes // 2, 1, 1, 0, bias=False),
                Norm_layer(inplanes // 2)
            )
        if down:
            factor = 2 if up else factor
            self.down = nn.Sequential(
                swish(),
                nn.Conv3d(inplanes, inplanes * 2, 1, factor, 0, bias=False),
                Norm_layer(inplanes * 2)
            )

        # Two separabe convolutions
        self.core = nn.Sequential(
            swish(),
            SeparableConv3d(inplanes, inplanes),
            Norm_layer(inplanes),
            swish(),
            SeparableConv3d(inplanes, inplanes),
            Norm_layer(inplanes)
        )

    def forward(self, x):
        out = [None, None, None]
        cell = self.core(x)
        if self.up is not None:
            out[0] = self.up(x + cell)
            out[0] = F.interpolate(
                out[0], scale_factor=self.factor, mode='trilinear',
                align_corners=True)
        if self.same:
            out[1] = x + cell  # Separables + residual
        if self.down is not None:
            out[2] = self.down(x + cell)
        return out


# MAKE IT PRETTIER!
class Layer(nn.Module):
    def __init__(self, depth, factor=2):
        super(Layer, self).__init__()
        self.depth = depth
        levels = [0] if (depth % 2 == 0) else [1]
        if depth < 7 and depth > 1:
            levels.append(levels[0] + 2)

        # I know, this is ugly...
        inputs = [[1, 2, 2, 2, 2],
                  [1, 3, 3, 3, 0],
                  [0, 1, 3, 3, 0],
                  [0, 1, 2, 0, 0]]
        c_factor = 2

        cells = []
        for lv in levels:
            up, keep, down = True, True, True
            channels = 64 * (2 ** lv)
            idx = depth // 2
            if idx < 4 and lv == 0:
                up = False
            if idx == 4 or (idx < 4 and inputs[lv][idx + 1] == 0):
                keep = False
                down = False
            elif lv == 3:
                down = False

            if lv < 2 and idx < 4:
                c_factor = factor[2]
            elif lv == 0 and idx == 4:
                c_factor = factor[1]
            else:
                c_factor = 2
            cells.append(Cell(channels, up, keep, down, c_factor))
        self.cells = nn.ModuleList(cells)

    def forward(self, x, initial_idx):
        for idx_c, cll in enumerate(self.cells):
            index = initial_idx + (idx_c * 2)
            in_cell = sum(x[index])
            x[index] = []  # Restart level
            out_cell = cll(in_cell)
            for idx_i, individual in enumerate(out_cell):
                if individual is not None:
                    x[(idx_i - 1) + index].append(individual)
        return x


class Network(nn.Module):
    def __init__(self, modalities, num_classes,
                 strides=[[2, 2, 1], [2, 2, 1], [2, 2, 2]]):
        super(Network, self).__init__()
        stem_1 = nn.Sequential(
            SeparableConv3d(modalities, 32),
            Norm_layer(32),
            swish(),
            SeparableConv3d(32, 32, stride=strides[0]),
            Norm_layer(32),
            swish()
        )
        stem_2 = nn.Sequential(
            SeparableConv3d(32, 32),
            Norm_layer(32),
            swish(),
            SeparableConv3d(32, 64, stride=strides[1]),
            Norm_layer(64)
        )
        self.stem = nn.ModuleList([stem_1, stem_2])

        backbone = []
        for depth in range(9):
            backbone.append(Layer(depth, strides))
        self.backbone = nn.ModuleList(backbone)

        self.combination = nn.Sequential(
                swish(),
                nn.Conv3d(32, 32, 3, 1, 1, bias=False),
                Norm_layer(32)
            )
        self.factor = strides[0]
        self.final = nn.Sequential(
                swish(),
                nn.Conv3d(32, num_classes, 1, 1, 0, bias=False),
            )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, a=1e-2)
                if m.bias is not None:
                    m.bias = nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x_small = self.stem[0](x)
        x = self.stem[1](x_small)

        x = [[x], [], [], []]
        for depth, layer in enumerate(self.backbone):
            initial_idx = 0 if depth % 2 == 0 else 1
            x = layer(x, initial_idx)
        x = x[-1][0] + x_small
        x = self.combination(x)
        x = F.interpolate(
            x, scale_factor=self.factor, mode='trilinear', align_corners=True)
        x = self.final(x)
        return x


class ROG(nn.Module):
    def __init__(self, params):
        super(ROG, self).__init__()
        self.ROG = Network(
            params['modalities'], params['classes'], params['strides'][:3])

        print('Number of parameters: {}'.format(
            sum([p.data.nelement() for p in self.ROG.parameters()])))

    def forward(self, x):
        x = self.ROG(x)
        return x
