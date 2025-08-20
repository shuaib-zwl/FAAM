import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.ops import DeformConv2dPack
from mmdet.models import NECKS
BN_MOMENTUM = 0.1


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for cc in range(1, w.size(0)):
        w[cc, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho, BN_MOMENTUM=0.1):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DeformConv2dPack(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    """IDAUp模块对多层特征进行上采样与融合，包含DeformConv，ConvTranspose2d以及node模块。
       这里增加了AFF(FFB)融合模块作为可选使用。"""
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        # o为输出通道数
        # channels为输入特征层通道列表
        # up_f为对应的上采样系数列表(通常是2的幂)
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

            # 如果需要AFF模块，可在此添加
            # setattr(self, 'aff_' + str(i), AFF(o))

    def forward(self, layers, startp, endp):
        # 按照从高层到低层的顺序进行上采样与融合
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            node = getattr(self, 'node_' + str(i - startp))

            layers[i] = upsample(project(layers[i]))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    """DLAUp模块使用多个IDAUp实例，对多尺度特征进行逐级融合上采样。
       输入的layers通常来自DLA的多层特征输出，如[x0, x1, x2, x3, x4]。
    """
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            ida = IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j])
            setattr(self, 'ida_{}'.format(i), ida)
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


@NECKS.register_module()
class DLANeck(nn.Module):
    def __init__(self, start_level=0, end_level=3, channels=[16, 32, 64, 128, 256], scales=None):
        super(DLANeck, self).__init__()
        self.first_level = start_level
        self.last_level = end_level
        if scales is None:
            # 通常scales根据channels个数自动生成,例如[1,2,4,8,16]
            scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        # 在DLAUp完成后，我们通常只需要在IDAUp的最后一层做额外融合
        out_channel = channels[self.first_level]
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

    def forward(self, features):
        fused_layers = self.dla_up(features)  # 将多尺度特征整合
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(fused_layers[i].clone())
        self.ida_up(y, 0, len(y))
        # head需要输入的是list，需要将变为list
        return [y[-1]]  # 返回最终融合后的特征图
