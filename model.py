# Copyright 2023 AlphaBetter Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
定义DefectGAN模型结构，结构来源于论文《Defect-GAN: High-Fidelity Defect Synthesis for Automated Defect Inspection》
论文链接：https://arxiv.org/abs/2103.15158

其中包含生成器，鉴别器和一个GP损失函数。
生成器主要类似于StarGAN，但是在生成器的最后一层使用了一个卷积层，而不是使用一个全连接层。
鉴别器主要是一个PatchGAN，用于判断输入的图像是否是真实的图像。
对于鉴别器增加了一个模型分类器，用于判断输入的图像类型。
"""
import math
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision.transforms import functional as F_vision

from common_types import _size_2_t
from utils import add_sn_

__all__ = [
    "DefectNet", "PathDiscriminator", "GradientPenaltyLoss",
    "defectnet", "path_discriminator", "gradient_penalty_loss"
]


class DefectNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_blocks: int = 6,
            num_down_blocks: int = 3,
            noise_image_size: int = 224,
            num_spatial_layers: int = 2,
    ) -> None:
        """DefectGAN的生成器

        Args:
            in_channels (int, optional): 输入的通道数，默认: 3
            out_channels (int, optional): 输出的通道数，默认: 3
            channels (int, optional): 卷积层的通道数，默认: 64
            num_blocks (int, optional): 残差块的数量，默认: 6
            num_down_blocks (int, optional): 下采样的残差块的数量，默认: 3
            noise_image_size (int, optional): 输入噪声图像的大小，默认: 224
            num_spatial_layers (int, optional): 空间图数量，如果是前景和背景则为2，默认: 2
        """

        super(DefectNet, self).__init__()
        noise_dim = (in_channels, noise_image_size, noise_image_size)
        self.adaptive_noise_mul = _AdaptiveNoiseMultiplier(noise_dim)

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels + noise_dim[0], channels, (7, 7), (1, 1), (3, 3), bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(False),
        )

        down_sampling = []
        current_channels = channels
        for _ in range(num_down_blocks):
            down_sampling.append(_Conv2dSamePadding(current_channels, int(current_channels * 2), kernel_size=4, stride=2, padding=0, bias=False))
            down_sampling.append(nn.InstanceNorm2d(int(current_channels * 2)))
            down_sampling.append(nn.ReLU(False))
            current_channels = current_channels * 2
        self.down_sampling = nn.Sequential(*down_sampling)

        trunk = []
        for _ in range(num_blocks):
            trunk.append(_IdentifyBlock(int(channels * (2 ** num_down_blocks)), int(channels * (2 ** num_down_blocks)), bias=False))
        self.trunk = nn.Sequential(*trunk)

        current_channels = channels * (2 ** num_down_blocks)
        self.up_conv1 = nn.ConvTranspose2d(current_channels, current_channels // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_spatial_block1 = _SpatialResConvBlock(num_spatial_layers, current_channels // 2, True)

        self.up_conv2 = nn.ConvTranspose2d(current_channels // 2, current_channels // 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_spatial_block2 = _SpatialResConvBlock(num_spatial_layers, current_channels // 4, True)

        self.up_conv3 = nn.ConvTranspose2d(current_channels // 4, current_channels // 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_spatial_block3 = _SpatialResConvBlock(num_spatial_layers, current_channels // 8, True)
        self.relu = nn.ReLU(False)

        self.overlay = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

        self.mask = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, spatial: Tensor, noise: Tensor) -> Tuple[Tensor, Tensor]:
        lambda_noise = self.adaptive_noise_mul(noise)
        x = torch.cat((x, lambda_noise[:, :, None, None] * noise), dim=1)

        x = self.first_layer(x)
        x = self.down_sampling(x)
        x = self.trunk(x)

        x = self.up_conv1(x)
        x = self.up_spatial_block1(spatial, x)
        x = self.relu(x)

        x = self.up_conv2(x)
        x = self.up_spatial_block2(spatial, x)
        x = self.relu(x)

        x = self.up_conv3(x)
        x = self.up_spatial_block3(spatial, x)
        x = self.relu(x)

        overlay = self.overlay(x)
        mask = self.mask(x)

        return overlay, mask


class PathDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
            num_blocks: int = 6,
            image_size: int = 224,
            num_classes: int = 2,
    ) -> None:
        """DefectGAN论文提出的鉴别器架构

        Args:
            in_channels (int, optional): 输入的通道数，默认: 3
            out_channels (int, optional): 输出的通道数，默认: 1
            channels (int, optional): 卷积层的通道数，默认: 64
            num_blocks (int, optional): 残差块的数量，默认: 6
            image_size (int, optional): 输入图像的大小，默认: 224
            num_classes (int, optional): 分类的数量，默认: 2
        """

        super(PathDiscriminator, self).__init__()
        if out_channels != 1:
            raise ValueError("out_channels must be 1")

        if image_size == 224:
            disc_feature_size = 9
        elif image_size == 128:
            disc_feature_size = 4
        else:
            raise ValueError("image_size must be 224 or 128")

        main = [
            _Conv2dSamePadding(in_channels, channels, kernel_size=4, stride=2, padding=0, bias=True),
            nn.ReLU(False)
        ]

        curr_channels = channels
        for _ in range(1, num_blocks):
            main.append(_Conv2dSamePadding(curr_channels, int(curr_channels * 2), kernel_size=4, stride=2, padding=0, bias=True))
            main.append(nn.LeakyReLU(0.01, False))
            curr_channels = int(curr_channels * 2)
        self.main = nn.Sequential(*main)

        self.disc = nn.Sequential(
            _Conv2dSamePadding(curr_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Linear(disc_feature_size, out_channels)
        )

        classifier_kernel = int(image_size / np.power(2, num_blocks))
        self.classifier = nn.Sequential(
            nn.Conv2d(curr_channels, num_classes, kernel_size=classifier_kernel, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.ReLU(False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.main(x)

        disc_output = self.disc(x)
        classifier_output = self.classifier(x)

        return disc_output, classifier_output


class _AdaptiveNoiseMultiplier(nn.Module):
    """对噪声进行控制，以便在不同的图像区域生成不同形状的噪声

    Args:
        noise_dim (tuple[int, int, int], optional): 噪声的维度，即噪声的形状，形状为(C, H, W)
        noise_channels (int, optional): 噪声的通道数
        channels (int, optional): 卷积层的通道数，默认: 32

    Examples:
        >>> x = torch.randn(1, 3, 224, 224)
        >>> m = _AdaptiveNoiseMultiplier()
        >>> out = m(x)
        >>> print(out.shape)
        >>> torch.Size([1, 1])
    """

    def __init__(self, noise_dim: tuple[int, int, int], noise_channels: int = 1, channels: int = 32) -> None:
        super(_AdaptiveNoiseMultiplier, self).__init__()
        self.adaptive_noise_multiplier = nn.Sequential(
            _Conv2dSamePadding(noise_dim[0], channels, kernel_size=5, stride=2, padding=0),
            _Conv2dSamePadding(channels, int(channels * 2), kernel_size=3, stride=2, padding=0),
            nn.ReLU(False),
            nn.Flatten(),
            nn.Linear(int(channels * 2) * np.prod(noise_dim[1:3]) // (4 ** 2), int(channels * 4)),
            nn.Linear(int(channels * 4), noise_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.adaptive_noise_multiplier(x)

        return x


class _Conv2dSamePadding(nn.Conv2d):
    """如果想要使卷积的形状更可预测，则使用此卷积方式替换原始Conv2D

    Args:
       参考nn.Conv2d

    Examples:
        >>> x = _Conv2dSamePadding(3, 6, 100, 3)(torch.randn(10, 3, 300, 300))
        >>> print(x.shape)
        >>> torch.Size([10, 6, 100, 100])
    """

    def __init__(self, *args, **kwargs) -> None:
        super(_Conv2dSamePadding, self).__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        padding = self._get_padding_for_same(self.kernel_size, self.stride, self.padding, x)
        x = self._conv_forward(F_torch.pad(x, padding), self.weight, self.bias)

        return x

    @staticmethod
    def _get_padding_for_same(kernel_size: _size_2_t or int,
                              stride: _size_2_t or int,
                              padding: _size_2_t or int,
                              x: Tensor):
        """计算padding的大小，使得卷积后的形状与卷积前一致

        Args:
            kernel_size (int or tuple): 卷积核大小
            stride (int or tuple): 卷积步长
            padding (int or tuple): 卷积padding
            x (Tensor): 输入的Tensor
        """
        if isinstance(kernel_size, int):
            # 如果kernel_size是int，则转换为tuple
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            # 如果stride是int，则转换为tuple
            stride = (stride, stride)
        if isinstance(padding, int):
            # 计算padding的大小
            x = F_torch.pad(x, (padding, padding, padding, padding))

        kernel_height = kernel_size[0]
        kernel_width = kernel_size[1]
        stride_height = stride[0]
        stride_width = stride[1]

        _, _, height, width = x.shape
        floor_height = math.floor(height / stride_height)
        floor_width = math.floor(width / stride_width)

        # 根据卷积的基本原理，计算出使输出特征图尺寸匹配的输入特征图需要添加的零填充量
        padding_width = (floor_width - 1) * stride_width + (kernel_width - 1) + 1 - width
        padding_height = (floor_height - 1) * stride_height + (kernel_height - 1) + 1 - height
        padding = (padding_width // 2, padding_width - padding_width // 2, padding_height // 2, padding_height - padding_height // 2)
        return padding


class _IdentifyBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        """基于InstanceNorm的残差块

        Args:
            in_channels (int): 输入的通道数
            out_channels (int): 输出的通道数
            bias (bool, optional): 卷积层是否使用偏置，默认: ``True``
        """
        super(_IdentifyBlock, self).__init__()
        self.identity_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same", bias=bias),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same", bias=bias),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.identity_block(x)

        x = x + identity

        return x


class _SpatialConv(nn.Module):
    def __init__(self, num_spatial_layers: int, channels: int, bias: bool = True) -> None:
        """空间卷积

        Args:
            num_spatial_layers (int): 空间层数，如果是前景和背景则为2
            channels (int): 卷积层的通道数
            bias (bool, optional): 卷积层是否使用偏置，默认: True
        """

        super(_SpatialConv, self).__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_spatial_layers, 128, kernel_size=5, stride=1, padding=2, bias=bias),
            nn.ReLU(False)
        )
        self.gamma = nn.Conv2d(128, channels, kernel_size=5, stride=1, padding=2, bias=bias)
        self.beta = nn.Conv2d(128, channels, kernel_size=5, stride=1, padding=2, bias=bias)

    def forward(self, segment_map: Tensor, x_in: Tensor) -> Tensor:
        # 对输入的Tensor进行归一化
        x = self._param_free_norm(x_in)

        # 对分割图进行下采样，使其与输入的Tensor形状一致
        _, _, x_h, x_w = x_in.shape
        segment_map_down = F_vision.resize(segment_map, (x_h, x_w))
        segment_map_down = self.spatial_conv(segment_map_down)

        # 对下采样后的分割图进行归一化
        segment_map_gamma = self.gamma(segment_map_down)
        segment_map_beta = self.beta(segment_map_down)

        # 对输入的Tensor进行反归一化
        x = x * (1 + segment_map_gamma) + segment_map_beta

        return x

    @staticmethod
    def _param_free_norm(x: Tensor, epsilon: float = 1e-5) -> Tensor:
        """对输入的Tensor进行归一化

        Args:
            x (Tensor): 输入的Tensor
            epsilon (float, optional): 防止除零错误的参数，默认: 1e-5
        """
        x_mean, x_var = torch.mean(x, dim=(2, 3)), torch.var(x, dim=(2, 3))
        x_std = torch.sqrt(x_var + epsilon)
        x = (x - x_mean[..., None, None]) * (1 / x_std)[..., None, None]

        return x


class _SpatialResConvBlock(nn.Module):
    def __init__(self, num_spatial_layers: int, channels: int, bias: bool = True) -> None:
        """空间残差卷积

        Args:
            num_spatial_layers (int): 空间层数，如果是前景和背景则为2
            channels (int, optional): 卷积层的通道数
            bias (bool, optional): 卷积层是否使用偏置，默认: True
        """
        super(_SpatialResConvBlock, self).__init__()
        self.spatial1 = _SpatialConv(num_spatial_layers, channels, bias)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.spatial2 = _SpatialConv(num_spatial_layers, channels, bias)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.leaky_relu = nn.LeakyReLU(0.2, False)

    def forward(self, segment_map, x) -> Tensor:
        out = self.spatial1(segment_map, x)
        out = self.leaky_relu(out)
        out = self.conv1(out)
        out = self.spatial2(segment_map, out)
        out = self.leaky_relu(out)
        out = self.conv2(out)

        out = out + x

        return out


class GradientPenaltyLoss(nn.Module):
    def __init__(self):
        """PyTorch实现GradientPenalty损失，以避免训练GAN过程中出现“模型崩塌”问题"""
        super(GradientPenaltyLoss, self).__init__()

    @staticmethod
    def forward(model: nn.Module, target: Tensor, source: Tensor) -> Tensor:
        BATCH_SIZE, C, H, W = target.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(target.device)
        interpolated_images = target * alpha + source * (1 - alpha)

        critic_inter, _ = model(interpolated_images)

        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=critic_inter,
            grad_outputs=torch.ones_like(critic_inter),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)

        return gradient_penalty


def defectnet(spectral_norm: bool = True, **kwargs) -> DefectNet:
    """DefectGAN的生成器

    Args:
        spectral_norm (bool, optional): 是否使用谱归一化，默认: ``True``
        **kwargs: 参考``DefectNet``

    Returns:
        DefectNet: DefectGAN的生成器
    """
    model = DefectNet(**kwargs)
    if spectral_norm:
        add_sn_(model)

    return model


def path_discriminator(spectral_norm: bool = True, **kwargs) -> PathDiscriminator:
    """DefectGAN的鉴别器

    Args:
        spectral_norm (bool, optional): 是否使用谱归一化，默认: ``True``
        **kwargs: 参考``PathDiscriminator``

    Returns:
        PathDiscriminator: DefectGAN的鉴别器
    """

    model = PathDiscriminator(**kwargs)
    if spectral_norm:
        add_sn_(model)

    return model


def gradient_penalty_loss() -> GradientPenaltyLoss:
    """PyTorch实现GradientPenalty损失，以避免训练GAN过程中出现“模型崩塌”问题

    Returns:
        GradientPenaltyLoss: PyTorch实现GradientPenalty损失
    """
    return GradientPenaltyLoss()
