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
import os
import warnings
from collections import OrderedDict
from enum import Enum
from typing import Any

import torch
import torch.backends.mps
from numpy import ndarray
from torch import nn, Tensor
from torch import distributed as dist
from torch.optim import Optimizer

__all__ = [
    "add_sn_", "load_state_dict", "load_pretrained_state_dict", "load_resume_state_dict", "make_directory",
    "get_sd_map_from_tensor", "swap_axes", "AverageMeter", "Summary", "ProgressMeter"
]


def _add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return nn.utils.spectral_norm(m)
    else:
        return m


def add_sn_(model: nn.Module):
    model.apply(_add_sn)


def load_state_dict(
        model: nn.Module,
        state_dict: dict,
        compile_mode: bool = False,
) -> nn.Module:
    """加载模型参数权重

    Args:
        model (nn.Module): PyTorch模型
        state_dict (dict): 模型权重和参数
        compile_mode (bool, optional): PyTorch2.0支持模型编译, 编译模型会比原始模型参数多一个前缀, 默认: ``False``

    Returns:
        model (nn.Module): 加载模型权重后的PyTorch模型
    """

    # 当PyTorch小于2.0时，不支持模型编译
    if int(torch.__version__[0]) < 2 and compile_mode:
        warnings.warn("PyTorch version is less than 2.0, does not support model compilation.")
        compile_mode = False

    # PyTorch2.0支持模型编译后, 编译模型会比原始模型参数多一个前缀，名为如下
    compile_state = "_orig_mod"

    # 创建新权重字典
    model_state_dict = model.state_dict()
    new_state_dict = OrderedDict()

    # 循环加载每一层模型权重
    for k, v in state_dict.items():
        current_compile_state = k.split(".")[0]

        if current_compile_state == compile_state and not compile_mode:
            name = k[len(compile_state) + 1:]
        elif current_compile_state != compile_state and compile_mode:
            raise ValueError("The model is not compiled, but the weight is compiled.")
        else:
            name = k
        new_state_dict[name] = v
    state_dict = new_state_dict

    # 过滤掉尺寸不相符的权重
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}

    # 更新模型权重
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    return model


def load_pretrained_state_dict(
        model: nn.Module,
        model_weights_path: str,
        compile_mode: bool = False,
) -> nn.Module:
    """加载预训练模型权重方法

    Args:
        model (nn.Module): PyTorch模型
        model_weights_path (str): model weights path
        compile_mode (bool, optional): PyTorch2.0支持模型编译, 编译模型会比原始模型参数多一个前缀, 默认: ``False``

    Returns:
        model (nn.Module): 加载模型权重后的PyTorch模型
    """

    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    state_dict = checkpoint["state_dict"]
    model = load_state_dict(model, state_dict, compile_mode)
    return model


def load_resume_state_dict(
        model: nn.Module,
        ema_model: Any,
        optimizer: Optimizer,
        scheduler: Any,
        model_weights_path: str,
        compile_mode: bool = False,
) -> Any:
    """恢复训练时候加载模型权重方法

    Args:
        model (nn.Module): model
        ema_model (nn.Module): EMA model
        optimizer (nn.optim): optimizer
        scheduler (nn.optim.lr_scheduler): learning rate scheduler
        model_weights_path (str): model weights path
        compile_mode (bool, optional): PyTorch2.0支持模型编译, 编译模型会比原始模型参数多一个前缀, 默认: ``False``

    Returns:
        model (nn.Module): 加载模型权重后的PyTorch模型
        ema_model (nn.Module): 加载经过EMA处理后的PyTorch模型
        start_epoch (int): 起始训练Epoch数
        optimizer (nn.optim): PyTorch优化器
        scheduler (nn.optim.lr_scheduler): PyTorch学习率调度器
    """

    # 加载模型权重
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

    # 提取模型权重中参数
    start_epoch = checkpoint["epoch"]
    state_dict = checkpoint["state_dict"]
    ema_state_dict = checkpoint["ema_state_dict"] if "ema_state_dict" in checkpoint else None

    model = load_state_dict(model, state_dict, compile_mode)
    if ema_state_dict is not None:
        ema_model = load_state_dict(ema_model, ema_state_dict, compile_mode)

    optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        scheduler = None

    return model, ema_model, start_epoch, optimizer, scheduler


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_sd_map_from_tensor(
        tensor: Tensor,
        num_spatial_classes: int = 2,
        tensor_shape: tuple = (224, 224),
        class_index: Tensor = 0,
) -> Tensor:
    """从掩码张量中获取空间分类张量

    Args:
        tensor (Tensor): 掩码张量
        num_spatial_classes (int, optional): 空间层数, 如果只有前景和背景, 则为2, 默认: 2
        tensor_shape (tuple, optional): 张量尺寸, 默认: (224, 224)
        class_index (Tensor, optional): 类别索引, 默认: 0
    """

    sd_map_tensor = torch.zeros(num_spatial_classes, tensor_shape[0], tensor_shape[1])
    sd_map_tensor[class_index] = tensor[0]

    return sd_map_tensor


def swap_axes(image: ndarray):
    image = image.swapaxes(0, 1).swapaxes(1, 2)

    return image


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.4f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.4f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.4f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
