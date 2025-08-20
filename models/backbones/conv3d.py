import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

class BasicConv3d(nn.Module):
    """Basic 3D Convolution Module."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@BACKBONES.register_module()
class Backbone3D(BaseModule):
    """Custom 3D Backbone based on the provided script."""
    def __init__(self,
                 in_channels=3,
                 channels=[16, 32, 64],
                 init_cfg=None):
        super(Backbone3D, self).__init__(init_cfg)
        # 3D卷积核的格式是[depth, H, W]
        # Branch 1
        self.branch1 = nn.Sequential(
            # BasicConv3d(in_channels, channels[0], kernel_size=(1, 1, 5), stride=(1, 1, 1), padding=(0, 0, 2)),
            BasicConv3d(in_channels, channels[0], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            # BasicConv3d(channels[0], channels[0], kernel_size=(1, 5, 1), stride=(1, 1, 1), padding=(0, 2, 0)),
            BasicConv3d(channels[0], channels[0], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(channels[0], channels[0], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        # Branch 2
        self.branch2 = nn.Sequential(
            BasicConv3d(channels[0], channels[1], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(channels[1], channels[1], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(channels[1], channels[1], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        # Branch 3
        self.branch3 = nn.Sequential(
            BasicConv3d(channels[1], channels[2], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(channels[2], channels[2], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(channels[2], channels[2], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        layers = []
        x = self.branch1(x)
        # save_first4_channels_3d(feature_map=x,save_dir='/home/ubuntu/data/ZWL/ZWL/data/pic/fmap_ch0-3/',cmap='viridis', per_channel_norm=True,percentile_clipping=None)
        layers.append(self.maxpool1(x).squeeze(2))  # First layer's feature map
        x = self.maxpool(x)
        x = self.branch2(x)
        layers.append(self.maxpool1(x).squeeze(2))  # Second layer's feature map
        x = self.maxpool(x)
        x = self.branch3(x)
        layers.append(self.maxpool1(x).squeeze(2))  # Third layer's feature map
        return layers  # Return multi-scale feature maps


def save_first4_channels_3d(
    feature_map,
    save_dir,
    cmap='viridis',
    per_channel_norm=True,
    percentile_clipping=(1, 99)
):
    """
    保存形状为 (1, C, T, H, W) 的 3D-CNN 特征图中前 4 个通道：
    每帧每通道生成一张原分辨率的 PNG。

    参数：
    - feature_map (torch.Tensor or np.ndarray)：输入特征图，形状 (1, C, T, H, W)。
    - save_dir (str)：保存目录，会自动创建。
    - cmap (str)：colormap。
    - per_channel_norm (bool)：是否对每个通道单独归一化到 [0,1]。
    - percentile_clipping (tuple or None)：(low_pct, high_pct) 百分位截断，None 表示不截断。
    """
    # 准备数据
    fmap = feature_map.detach().cpu().numpy() if isinstance(feature_map, torch.Tensor) else np.array(feature_map)
    assert fmap.ndim == 5 and fmap.shape[0] == 1, f"Expected shape (1,C,T,H,W), got {fmap.shape}"
    _, C, T, H, W = fmap.shape
    fmap = fmap.squeeze(0)  # (C, T, H, W)

    os.makedirs(save_dir, exist_ok=True)

    # 仅限通道 0–3
    for c in range(min(4, C)):
        for t in range(T):
            channel = fmap[c, t]

            # 百分位截断
            if percentile_clipping is not None:
                low, high = np.percentile(channel, percentile_clipping)
                channel = np.clip(channel, low, high)

            # 每通道归一化
            if per_channel_norm:
                ch_min, ch_max = channel.min(), channel.max()
                if ch_max > ch_min:
                    channel = (channel - ch_min) / (ch_max - ch_min)

            # 保存：直接以原 H×W 分辨率输出
            out_path = os.path.join(
                save_dir,
                f'frame{t}_ch{c}.png'
            )
            plt.imsave(out_path, channel, cmap=cmap)
            print(f"Saved: {out_path}")

