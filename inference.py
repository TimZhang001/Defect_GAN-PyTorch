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
from matplotlib import pyplot as plt

from official_dataset import DefectDataset
from dataset import DefectGANDataset
from model import defectnet
from utils import swap_axes
from torch.utils import data


def view_intermediate_result(gen, image_name):
    plt.figure(figsize=(17, 10))
    for i in range(0, 1):
        with torch.no_grad():
            data_gen = iter(defectDataloader)

            # Official
            # defect_cls_index, normals, defects, defect_mask, spatial_cat_maps = next(data_gen)
            #
            # spatial_cat_map = spatial_cat_maps[i]
            # normal = normals[i]
            # defect = defects[i]
            #
            # sd_map = spatial_cat_maps[i]
            # normal = normals[i]

            # Ours
            batch_data = next(data_gen)
            defect_cls_index = batch_data["class_index"]
            defect = batch_data["defect_image_tensor"].squeeze(0)
            sd_map = batch_data["sd_map_tensor"].squeeze(0)

            gen.eval()

            for row_count, (from_, spa_cat) in enumerate([(defect, sd_map)]):
                z = torch.ones((1, 3, 224, 224), dtype=torch.float)

                top, m = gen(from_[None, ...], spa_cat[None, ...], z)

                top_layer = swap_axes(np.squeeze(top.cpu().numpy()))
                m = swap_axes(m.cpu().numpy()[0])

                defect_index = int(defect_cls_index[i])

                from_ = swap_axes(from_.cpu().numpy())
                spa_cat = swap_axes(spa_cat.cpu().numpy())

                plt.subplot(3, 5, 5 * row_count + 1)
                plt.imshow(((from_ + 1) * 127.5).astype("uint8"))

                plt.subplot(3, 5, 5 * row_count + 2)
                plt.imshow(spa_cat[:, :, defect_index])

                plt.subplot(3, 5, 5 * row_count + 3)

                plt.imshow(((top_layer + 1) * 127.5).astype("uint8"))

                plt.subplot(3, 5, 5 * row_count + 4)
                plt.imshow(np.squeeze(m))

                plt.subplot(3, 5, 5 * row_count + 5)
                gened_defects = from_ * (1 - m) + top_layer * m
                plt.imshow(((gened_defects + 1) * 127.5).astype("uint8"))

            plt.savefig(image_name,
                        dpi=80,
                        bbox_inches="tight")

            gen.train()


if __name__ == "__main__":
    # defect_dataset = DefectDataset()
    defect_dataset = DefectGANDataset("./data/ours_datasets")
    defectDataloader = data.DataLoader(dataset=defect_dataset,
                                       batch_size=1,
                                       shuffle=False,
                                       drop_last=True,
                                       num_workers=4)

    gen = defectnet()
    gen.load_state_dict(torch.load("../Defect_GAN-SRC/checkpoints/gen_epoch_0_batch_500.pt")["state_dict"])

    view_intermediate_result(gen, "ours_datasets.jpg")
