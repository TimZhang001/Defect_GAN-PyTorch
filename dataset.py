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
import queue
import threading

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from utils import get_sd_map_from_tensor

__all__ = [
    "DefectGANDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


class DefectGANDataset(data.Dataset):
    def __init__(
            self,
            root: str,
            image_size: int = 224,
            num_classes: int = 2,
            device_id: str = "cpu",
    ) -> None:
        """DefectGAN中缺陷数据集加载类

        Args:
            root (str): 缺陷数据集根目录，根目录下应该有三个子目录，分别是normal、defect和defect_mask
            image_size (int, optional): 图像大小, 默认: 224
            num_classes (int, optional): 缺陷类别数，这里只分为OK或者NG，所以是2类, 默认: 2
            device_id (str, optional): 设备ID, 可以是"cpu"或者一个非负整数, 默认: ``cpu``
        """

        super(DefectGANDataset, self).__init__()
        self.root = root
        self.image_size = image_size
        self.num_classes = num_classes
        self.device_id = device_id

        self.defect_dataset_list = []

        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.image_mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        if device_id == "cpu":
            self.device = torch.device("cpu")
        elif int(device_id) >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda", int(device_id))
        else:
            raise ValueError("Device ID should be 'cpu' or a non-negative integer.")

        self.load_all_image_file_paths()

    def load_all_image_file_paths(self):
        if len(self.defect_dataset_list) == 0:
            normal_dataset_dir = f"{self.root}/normal"
            defect_dataset_dir = f"{self.root}/defect"
            defect_mask_dataset_dir = f"{self.root}/defect_mask"

            if not os.path.exists(normal_dataset_dir):
                raise FileNotFoundError(f"Normal dataset directory {normal_dataset_dir} does not exist. "
                                        f"You should set normal dataset directory name is `normal`.")
            if not os.path.exists(defect_dataset_dir):
                raise FileNotFoundError(f"Defect dataset directory {defect_dataset_dir} does not exist. "
                                        f"You should set defect dataset directory name is `defect`.")
            if not os.path.exists(defect_mask_dataset_dir):
                raise FileNotFoundError(f"Defect mask dataset directory {defect_mask_dataset_dir} does not exist. "
                                        f"You should set defect mask dataset directory name is `defect_mask`.")

            for image_file_path in os.listdir(normal_dataset_dir):
                normal_image_path = f"{normal_dataset_dir}/{image_file_path}"
                defect_image_path = f"{defect_dataset_dir}/{image_file_path}"
                defect_mask_path = f"{defect_mask_dataset_dir}/{image_file_path}"
                self.defect_dataset_list.append([1, normal_image_path, defect_image_path, defect_mask_path])

    def load_image_class_and_image_path_from_index(self, batch_index: int):
        class_index, normal_image_path, defect_image_path, defect_mask_path = self.defect_dataset_list[batch_index]
        normal_tensor = self.image_transform(Image.open(normal_image_path)).to(self.device)
        defect_tensor = self.image_transform(Image.open(defect_image_path)).to(self.device)
        defect_mask_tensor = self.image_mask_transform(Image.open(defect_mask_path)).to(self.device)
        defect_mask_tensor = torch.where(defect_mask_tensor > 0.5, 1, 0)

        return class_index, normal_tensor, defect_tensor, defect_mask_tensor

    def load_sd_map_from_index(self, batch_index: int):
        class_index, _, _, defect_mask_tensor = self.load_image_class_and_image_path_from_index(batch_index)
        sd_map = get_sd_map_from_tensor(defect_mask_tensor, self.num_classes, (self.image_size, self.image_size), class_index)

        return sd_map

    def __getitem__(self, batch_index: int):
        class_index, normal_tensor, defect_tensor, defect_mask_tensor = self.load_image_class_and_image_path_from_index(batch_index)
        sd_map_tensor = self.load_sd_map_from_index(batch_index)

        return {
            "class_index": torch.as_tensor(class_index).type(torch.LongTensor),
            "normal_tensor": normal_tensor,
            "defect_tensor": defect_tensor,
            "defect_mask_tensor": defect_mask_tensor,
            "sd_map_tensor": sd_map_tensor
        }

    def __len__(self):
        return len(self.defect_dataset_list)


class PrefetchGenerator(threading.Thread):
    """借助PyTorch队列功能生成数据生成器

    Args:
        generator (Generator): 生成器
        num_data_prefetch_queue (int): 预加载数据队列长度
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(data.DataLoader):
    """借助PyTorch队列功能生成DataLoader预加载器

    Args:
        num_data_prefetch_queue (int): 预加载数据队列长度
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """CPU版本的数据预加载器

    Args:
        dataloader (DataLoader): PrefetchDataLoader预加载器
    """

    def __init__(self, dataloader: data.DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """CUDA版本的数据预加载器

    Args:
        dataloader (DataLoader): PrefetchDataLoader预加载器
        device (torch.device): 设备类型
    """

    def __init__(self, dataloader: data.DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
