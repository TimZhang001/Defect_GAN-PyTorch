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
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from model import defectnet
from utils import add_sn_, load_pretrained_state_dict, get_sd_map_from_tensor, swap_axes

if __name__ == "__main__":
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_mask_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    class_index = torch.tensor([0])

    normal_tensor = image_transform(Image.open("./defect4.png")).unsqueeze(0)
    defect_mask_tensor = image_mask_transform(Image.open("./normal_mask.png")).unsqueeze(0)
    sd_map = get_sd_map_from_tensor(defect_mask_tensor, 2, (224, 224), class_index)
    noise = torch.randn((1, 3, 224, 224), dtype=torch.float)
    gen = defectnet()

    # ==========================================
    # Official
    gen.load_state_dict(torch.load("../Defect_GAN-SRC/checkpoints/gen_epoch_9_batch_500.pt")["state_dict"])
    top, m = gen(normal_tensor, sd_map, noise)

    top_layer = swap_axes(np.squeeze(top.cpu().detach().numpy()))
    m = swap_axes(m.cpu().detach().numpy()[0])

    defect_index = int(class_index[0])

    from_ = swap_axes(normal_tensor.squeeze(0).cpu().numpy())
    spa_cat = swap_axes(sd_map.squeeze(0).cpu().numpy())

    plt.subplot(3, 5, 1)
    plt.imshow(((from_ + 1) * 127.5).astype("uint8"))

    plt.subplot(3, 5, 2)
    plt.imshow(spa_cat[:, :, defect_index])

    plt.subplot(3, 5, 3)
    plt.imshow(((top_layer + 1) * 127.5).astype("uint8"))
    plt.imsave("official_overlap.png", ((top_layer + 1) * 127.5).astype("uint8"))

    plt.subplot(3, 5, 4)
    plt.imshow(np.squeeze(m))

    plt.subplot(3, 5, 5)
    gened_defects = from_ * (1 - m) + top_layer * m
    plt.imshow(((gened_defects + 1) * 127.5).astype("uint8"))
    plt.imsave("official_results.png", ((gened_defects + 1) * 127.5).astype("uint8"))

    plt.savefig("official_visual.jpg", dpi=80, bbox_inches="tight")

    # ==========================================
    # Ours
    ge = torch.compile(gen)
    gen = load_pretrained_state_dict(gen, "./samples/defect_gan-official-20230926-15_47_23/g_epoch_3.pth.tar")
    top, m = gen(normal_tensor, sd_map, noise)

    top_layer = swap_axes(np.squeeze(top.cpu().detach().numpy()))
    m = swap_axes(m.cpu().detach().numpy()[0])

    defect_index = int(class_index[0])

    from_ = swap_axes(normal_tensor.squeeze(0).cpu().numpy())
    spa_cat = swap_axes(sd_map.squeeze(0).cpu().numpy())

    plt.subplot(3, 5, 1)
    plt.imshow(((from_ + 1) * 127.5).astype("uint8"))

    plt.subplot(3, 5, 2)
    plt.imshow(spa_cat[:, :, defect_index])

    plt.subplot(3, 5, 3)
    plt.imshow(((top_layer + 1) * 127.5).astype("uint8"))
    plt.imsave("ours_overlap.png", ((top_layer + 1) * 127.5).astype("uint8"))

    plt.subplot(3, 5, 4)
    plt.imshow(np.squeeze(m))

    plt.subplot(3, 5, 5)
    gened_defects = from_ * (1 - m) + top_layer * m
    plt.imshow(((gened_defects + 1) * 127.5).astype("uint8"))
    plt.imsave("ours_results.png", ((gened_defects + 1) * 127.5).astype("uint8"))

    plt.savefig("ours_visual.jpg", dpi=80, bbox_inches="tight")
