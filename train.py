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
import argparse
import random
import time
import warnings
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim.swa_utils import AveragedModel
from torch.utils import data

import wandb
from dataset import DefectGANDataset, CPUPrefetcher, CUDAPrefetcher
from model import defectnet, path_discriminator, gradient_penalty_loss
from utils import load_pretrained_state_dict, load_resume_state_dict, make_directory, AverageMeter, Summary, ProgressMeter


class Trainer(object):
    def __init__(self, config: Any):
        # 运行环境相关参数
        self.project_name = config["PROJECT_NAME"]
        self.exp_name = config["EXP_NAME"] + time.strftime("-%Y%m%d-%H_%M_%S", time.localtime(int(round(time.time() * 1000)) / 1000))
        self.seed = config["SEED"]
        self.mixing_precision = config["MIXING_PRECISION"]
        self.scaler = None  # TODO: 未来支持混合精度训练
        self.device = config["DEVICE"]
        self.cudnn_benchmark = config["CUDNN_BENCHMARK"]

        self.wandb_config = config
        self.wandb_project_name = config["PROJECT_NAME"]

        self.samples_dir = f"./samples/{self.exp_name}"
        self.results_dir = f"./results/{self.exp_name}"
        self.visuals_dir = f"./results/visuals/{self.exp_name}"

        # 模型相关参数
        self.g_model = None
        self.ema_g_model = None
        self.g_model_name = config["MODEL"]["G"]["NAME"]
        self.g_model_in_channels = config["MODEL"]["G"]["IN_CHANNELS"]
        self.g_model_out_channels = config["MODEL"]["G"]["OUT_CHANNELS"]
        self.g_model_channels = config["MODEL"]["G"]["CHANNELS"]
        self.g_model_num_blocks = config["MODEL"]["G"]["NUM_BLOCKS"]
        self.g_model_num_down_blocks = config["MODEL"]["G"]["NUM_DOWN_BLOCKS"]
        self.g_model_noise_image_size = config["MODEL"]["G"]["NOISE_IMAGE_SIZE"]
        self.g_model_num_spatial_layers = config["MODEL"]["G"]["NUM_SPATIAL_LAYERS"]
        self.g_model_spectral_norm = config["MODEL"]["G"]["SPECTRAL_NORM"]
        self.g_model_ema = config["MODEL"]["G"]["EMA"]
        self.g_model_compiled = config["MODEL"]["G"]["COMPILED"]

        self.d_model = None
        self.ema_d_model = None
        self.d_model_name = config["MODEL"]["D"]["NAME"]
        self.d_model_in_channels = config["MODEL"]["D"]["IN_CHANNELS"]
        self.d_model_out_channels = config["MODEL"]["D"]["OUT_CHANNELS"]
        self.d_model_channels = config["MODEL"]["D"]["CHANNELS"]
        self.d_model_num_blocks = config["MODEL"]["D"]["NUM_BLOCKS"]
        self.d_model_image_size = config["MODEL"]["D"]["IMAGE_SIZE"]
        self.d_model_num_classes = config["MODEL"]["D"]["NUM_CLASSES"]
        self.d_model_spectral_norm = config["MODEL"]["D"]["SPECTRAL_NORM"]
        self.d_model_ema = config["MODEL"]["D"]["EMA"]
        self.d_model_compiled = config["MODEL"]["D"]["COMPILED"]

        self.ema_avg_fn = None
        self.ema_decay = config["MODEL"]["EMA"]["DECAY"]
        self.ema_compiled = config["MODEL"]["EMA"]["COMPILED"]

        self.pretrained_g_model_weights_path = config["MODEL"]["CHECKPOINT"]["PRETRAINED_G_MODEL_WEIGHTS_PATH"]
        self.pretrained_d_model_weights_path = config["MODEL"]["CHECKPOINT"]["PRETRAINED_D_MODEL_WEIGHTS_PATH"]
        self.resumed_g_model_weights_path = config["MODEL"]["CHECKPOINT"]["RESUME_G_MODEL_WEIGHTS_PATH"]
        self.resumed_d_model_weights_path = config["MODEL"]["CHECKPOINT"]["RESUME_D_MODEL_WEIGHTS_PATH"]

        # 数据集相关参数
        self.train_root_dir = config["TRAIN"]["DATASET"]["ROOT_DIR"]
        self.train_batch_size = config["TRAIN"]["HYP"]["IMGS_PER_BATCH"]
        self.train_shuffle = config["TRAIN"]["HYP"]["SHUFFLE"]
        self.train_num_workers = config["TRAIN"]["HYP"]["NUM_WORKERS"]
        self.train_pin_memory = config["TRAIN"]["HYP"]["PIN_MEMORY"]
        self.train_drop_last = config["TRAIN"]["HYP"]["DROP_LAST"]
        self.train_persistent_workers = config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"]

        self.train_data_prefetcher = None

        # 损失函数参数
        self.rec_criterion = None
        self.cls_criterion = None
        self.gp_criterion = None

        self.rec_criterion_name = config["TRAIN"]["LOSSES"]["REC_CRITERION"]["NAME"]
        self.cls_criterion_name = config["TRAIN"]["LOSSES"]["CLS_CRITERION"]["NAME"]
        self.gp_criterion_name = config["TRAIN"]["LOSSES"]["GP_CRITERION"]["NAME"]

        self.g_gp_loss_weight = config["TRAIN"]["LOSSES"]["LAMBDA"]["G_GP_LOSS_WEIGHT"]
        self.g_fake_cls_loss_weight = config["TRAIN"]["LOSSES"]["LAMBDA"]["G_FAKE_CLS_LOSS_WEIGHT"]
        self.g_rec_loss_weight = config["TRAIN"]["LOSSES"]["LAMBDA"]["G_REC_LOSS_WEIGHT"]
        self.g_cycle_rec_loss_weight = config["TRAIN"]["LOSSES"]["LAMBDA"]["G_CYCLE_REC_LOSS_WEIGHT"]
        self.g_cycle_mask_rec_loss_weight = config["TRAIN"]["LOSSES"]["LAMBDA"]["G_CYCLE_MASK_REC_LOSS_WEIGHT"]
        self.g_cycle_mask_vanishing_loss_weight = config["TRAIN"]["LOSSES"]["LAMBDA"]["G_CYCLE_MASK_VANISHING_LOSS_WEIGHT"]
        self.g_cycle_spatial_loss_weight = config["TRAIN"]["LOSSES"]["LAMBDA"]["G_CYCLE_SPATIAL_LOSS_WEIGHT"]

        self.d_gp_loss_weight = config["TRAIN"]["LOSSES"]["LAMBDA"]["D_GP_LOSS_WEIGHT"]
        self.d_real_cls_loss_weight = config["TRAIN"]["LOSSES"]["LAMBDA"]["D_REAL_CLS_LOSS_WEIGHT"]

        # 优化器参数
        self.g_optimizer = None
        self.g_optimizer_name = config["TRAIN"]["OPTIMIZER"]["G"]["NAME"]
        self.g_optimizer_lr = config["TRAIN"]["OPTIMIZER"]["G"]["LR"]
        self.g_optimizer_betas = config["TRAIN"]["OPTIMIZER"]["G"]["BETAS"]
        self.d_optimizer = None
        self.d_optimizer_name = config["TRAIN"]["OPTIMIZER"]["D"]["NAME"]
        self.d_optimizer_lr = config["TRAIN"]["OPTIMIZER"]["D"]["LR"]
        self.d_optimizer_betas = config["TRAIN"]["OPTIMIZER"]["D"]["BETAS"]

        # 学习率调度器参数
        self.g_scheduler = None
        self.d_scheduler = None

        # 训练参数
        self.start_epoch = 0
        self.epochs = config["TRAIN"]["HYP"]["EPOCHS"]
        self.print_freq = config["TRAIN"]["PRINT_FREQ"]
        self.normal_label = 0
        self.defect_label = 1
        self.g_gp_loss = torch.Tensor([0.0])
        self.g_fake_cls_loss = torch.Tensor([0.0])
        self.g_rec_loss = torch.Tensor([0.0])
        self.g_cycle_rec_loss = torch.Tensor([0.0])
        self.g_cycle_mask_rec_loss = torch.Tensor([0.0])
        self.g_cycle_mask_vanishing_loss = torch.Tensor([0.0])
        self.g_cycle_spatial_loss = torch.Tensor([0.0])
        self.d_gp_loss = torch.Tensor([0.0])
        self.d_real_cls_loss = torch.Tensor([0.0])
        self.g_loss = torch.Tensor([0.0])
        self.d_loss = torch.Tensor([0.0])

        # 训练环境
        make_directory(self.samples_dir)
        make_directory(self.results_dir)
        make_directory(self.visuals_dir)
        self.setup_seed()
        self.setup_mixing_precision()
        self.setup_device()
        self.setup_wandb()
        # 模型
        self.build_models()
        # 数据集
        self.load_datasets()
        # 损失函数
        self.define_loss()
        # 优化器
        self.define_optimizer()
        # 学习率调度器
        self.define_scheduler()
        # 加载模型权重
        self.load_model_weights()

    def setup_seed(self):
        # 固定随机数种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def setup_mixing_precision(self):
        # 初始化混合精度训练方法
        if self.mixing_precision:
            self.scaler = amp.GradScaler()
        else:
            print("Mixing precision training is not enabled.")

    def setup_device(self):
        # 初始化训练的设备名称
        device = "cpu"
        if self.device != "cpu" and self.device != "":
            if not torch.cuda.is_available():
                warnings.warn("No GPU detected, defaulting to `cpu`.")
            else:
                device = self.device
        if self.device == "":
            warnings.warn("No device specified, defaulting to `cpu`.")
        self.device = torch.device(device)

        # 如果输入图像尺寸是固定的，固定卷积算法可以提升训练速度
        if self.cudnn_benchmark:
            cudnn.benchmark = True
        else:
            cudnn.benchmark = False

    def setup_wandb(self):
        # 初始化wandb
        wandb.init(config=self.wandb_config, project=self.wandb_project_name, name=self.exp_name)

    def build_models(self):
        if self.g_model_name == "defectnet":
            self.g_model = defectnet(spectral_norm=self.g_model_spectral_norm,
                                     in_channels=self.g_model_in_channels,
                                     out_channels=self.g_model_out_channels,
                                     channels=self.g_model_channels,
                                     num_blocks=self.g_model_num_blocks,
                                     num_down_blocks=self.g_model_num_down_blocks,
                                     noise_image_size=self.g_model_noise_image_size,
                                     num_spatial_layers=self.g_model_num_spatial_layers)
        else:
            raise ValueError(f"The `{self.g_model_name}` is not supported.")

        if self.d_model_name == "path_discriminator":
            self.d_model = path_discriminator(spectral_norm=self.d_model_spectral_norm,
                                              in_channels=self.d_model_in_channels,
                                              out_channels=self.d_model_out_channels,
                                              channels=self.d_model_channels,
                                              num_blocks=self.d_model_num_blocks,
                                              image_size=self.d_model_image_size,
                                              num_classes=self.d_model_num_classes)
        else:
            raise ValueError(f"The `{self.d_model_name}` is not supported.")

        # 送至指定设备上运行
        self.g_model = self.g_model.to(self.device)
        self.d_model = self.d_model.to(self.device)

        if self.ema_decay != 0:
            self.ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
                (1 - self.ema_decay) * averaged_model_parameter + self.ema_decay * model_parameter
        if self.ema_g_model:
            self.ema_g_model = AveragedModel(self.g_model, device=self.device, avg_fn=self.ema_avg_fn)
        if self.ema_d_model:
            self.ema_d_model = AveragedModel(self.d_model, device=self.device, avg_fn=self.ema_avg_fn)

        # 编译模型
        if config["MODEL"]["G"]["COMPILED"]:
            self.g_model = torch.compile(self.g_model)
        if config["MODEL"]["D"]["COMPILED"]:
            self.d_model = torch.compile(self.d_model)
        if config["MODEL"]["EMA"]["COMPILED"]:
            if self.ema_g_model is not None:
                self.ema_g_model = torch.compile(self.ema_g_model)
            if self.ema_d_model is not None:
                self.ema_d_model = torch.compile(self.ema_d_model)
                warnings.warn("Dynamic compilation of discriminator is not recommended, "
                              "and the support on PyTorch2.0.1 version is not good enough.")

    def load_datasets(self):
        defect_dataset = DefectGANDataset(self.train_root_dir)
        defect_dataloader = data.DataLoader(
            defect_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.train_num_workers,
            pin_memory=self.train_pin_memory,
            drop_last=self.train_drop_last,
            persistent_workers=self.train_persistent_workers,
        )

        if self.device.type == "cuda":
            # 将数据加载器替换为CUDA以加速
            self.train_data_prefetcher = CUDAPrefetcher(defect_dataloader, self.device)
        if self.device.type == "cpu":
            # 将数据加载器替换为CPU以加速
            self.train_data_prefetcher = CPUPrefetcher(defect_dataloader)

    def define_loss(self):
        if self.rec_criterion_name == "l1":
            self.rec_criterion = nn.L1Loss()
        else:
            raise NotImplementedError(f"Loss {self.rec_criterion_name} is not supported.")
        if self.cls_criterion_name == "cross_entropy":
            self.cls_criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss {self.cls_criterion_name} is not supported.")
        if self.gp_criterion_name == "gradient_penalty":
            self.gp_criterion = gradient_penalty_loss()
        else:
            raise NotImplementedError(f"Loss {self.gp_criterion_name} is not supported.")

        self.rec_criterion = self.rec_criterion.to(self.device)
        self.cls_criterion = self.cls_criterion.to(self.device)
        self.gp_criterion = self.gp_criterion.to(self.device)

    def define_optimizer(self):
        if self.g_optimizer_name == "adam":
            self.g_optimizer = optim.Adam(self.g_model.parameters(),
                                          self.g_optimizer_lr,
                                          self.g_optimizer_betas)
        else:
            raise NotImplementedError(f"Optimizer {self.g_optimizer_name} is not supported.")
        if self.d_optimizer_name == "adam":
            self.d_optimizer = optim.Adam(self.d_model.parameters(),
                                          self.d_optimizer_lr,
                                          self.d_optimizer_betas)
        else:
            raise NotImplementedError(f"Optimizer {self.d_optimizer_name} is not supported.")

    def define_scheduler(self):
        pass

    def load_model_weights(self):
        if self.pretrained_g_model_weights_path != "":
            self.g_model = load_pretrained_state_dict(self.g_model, self.pretrained_g_model_weights_path, self.g_model_compiled)
            self.g_model = torch.load(self.pretrained_g_model_weights_path)
            print(f"Loaded `{self.pretrained_g_model_weights_path}` pretrained model weights successfully.")
        if self.pretrained_d_model_weights_path != "":
            self.d_model = load_pretrained_state_dict(self.d_model, self.pretrained_d_model_weights_path, self.d_model_compiled)
            print(f"Loaded `{self.pretrained_d_model_weights_path}` pretrained model weights successfully.")

        if self.resumed_g_model_weights_path != "":
            self.g_model, self.ema_g_model, self.start_epoch, self.g_optimizer, self.g_scheduler = load_resume_state_dict(
                self.g_model,
                self.ema_g_model,
                self.g_optimizer,
                self.g_scheduler,
                self.resumed_g_model_weights_path,
                self.g_model_compiled,
            )
            print(f"Loaded `{self.resumed_g_model_weights_path}` resume model weights successfully.")

        if self.resumed_d_model_weights_path != "":
            self.d_model, self.ema_d_model, self.start_epoch, self.d_optimizer, self.d_scheduler = load_resume_state_dict(
                self.d_model,
                self.ema_d_model,
                self.d_optimizer,
                self.d_scheduler,
                self.resumed_d_model_weights_path,
                self.d_model_compiled,
            )
            print(f"Loaded `{self.resumed_d_model_weights_path}` resume model weights successfully.")

    def train(self):
        # 将模型调整为训练模式
        self.g_model.train()
        self.d_model.train()

        # 用于生成器输入和重建时候噪声输入
        fake_noise = torch.randn(
            self.train_batch_size,
            self.g_model_in_channels,
            self.g_model_noise_image_size,
            self.g_model_noise_image_size).to(self.device)
        rec_noise = torch.randn(
            self.train_batch_size,
            self.g_model_in_channels,
            self.g_model_noise_image_size,
            self.g_model_noise_image_size).to(self.device)

        # 将正常,缺陷样本的标签设置为0，1
        normal_class_index = torch.as_tensor([self.normal_label] * self.train_batch_size).type(torch.LongTensor).to(self.device)
        defect_class_index = torch.as_tensor([self.defect_label] * self.train_batch_size).type(torch.LongTensor).to(self.device)

        # 损失函数权重
        self.g_gp_loss_weight = torch.Tensor([self.g_gp_loss_weight]).to(self.device)
        self.g_fake_cls_loss_weight = torch.Tensor([self.g_fake_cls_loss_weight]).to(self.device)
        self.g_rec_loss_weight = torch.Tensor([self.g_rec_loss_weight]).to(self.device)
        self.g_cycle_rec_loss_weight = torch.Tensor([self.g_cycle_rec_loss_weight]).to(self.device)
        self.g_cycle_mask_rec_loss_weight = torch.Tensor([self.g_cycle_mask_rec_loss_weight]).to(self.device)
        self.g_cycle_mask_vanishing_loss_weight = torch.Tensor([self.g_cycle_mask_vanishing_loss_weight]).to(self.device)
        self.g_cycle_spatial_loss_weight = torch.Tensor([self.g_cycle_spatial_loss_weight]).to(self.device)

        self.d_gp_loss_weight = torch.Tensor([self.d_gp_loss_weight]).to(self.device)
        self.d_real_cls_loss_weight = torch.Tensor([self.d_real_cls_loss_weight]).to(self.device)

        for epoch in range(self.start_epoch, self.epochs):
            batch_index = 1
            self.train_data_prefetcher.reset()
            end = time.time()
            batch_data = self.train_data_prefetcher.next()

            # 计算一个epoch的批次数量
            batches = len(self.train_data_prefetcher)
            # 进度条信息
            batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
            data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
            g_losses = AverageMeter("G loss", ":6.6f", Summary.NONE)
            d_losses = AverageMeter("D loss", ":6.6f", Summary.NONE)

            progress = ProgressMeter(batches,
                                     [batch_time, data_time, g_losses, d_losses],
                                     f"Epoch: [{epoch + 1}]")

            while batch_data is not None:
                # 计算加载一个批次数据时间
                data_time.update(time.time() - end)

                normals = batch_data["normal_tensor"]
                defects = batch_data["defect_tensor"]
                sd_maps = batch_data["sd_map_tensor"]

                self.train_batch(normals, sd_maps, normal_class_index, defect_class_index, fake_noise, rec_noise)  # Train norm to defect
                self.train_batch(defects, sd_maps, defect_class_index, normal_class_index, fake_noise, rec_noise)  # Train defect to normal

                # 统计需要打印的损失
                g_losses.update(self.g_loss.item(), self.train_batch_size)
                d_losses.update(self.d_loss.item(), self.train_batch_size)

                # 计算训练完一个批次时间
                batch_time.update(time.time() - end)
                end = time.time()

                current_iter = batch_index + epoch * batches + 1
                # 保存训练日志
                wandb.log({
                    "iter": current_iter,
                    "g_gp_loss": self.g_gp_loss,
                    "g_fake_cls_loss": self.g_fake_cls_loss,
                    "g_rec_loss": self.g_rec_loss,
                    "g_cycle_rec_loss": self.g_cycle_rec_loss,
                    "g_cycle_mask_rec_loss": self.g_cycle_mask_rec_loss,
                    "g_cycle_mask_vanishing_loss": self.g_cycle_mask_vanishing_loss,
                    "g_cycle_spatial_loss": self.g_cycle_spatial_loss,
                    "d_gp_loss": self.d_gp_loss,
                    "d_real_cls_loss": self.d_real_cls_loss,
                })
                # 打印训练进度
                if self.print_freq <= 0:
                    raise ValueError(f"Invalid value of print_freq: {self.print_freq}, must be greater than 0.")
                if batch_index == 0 or (batch_index + 1) % self.print_freq == 0:
                    progress.display(batch_index + 1)

                # 加载下一个batch_data
                batch_index += 1
                batch_data = self.train_data_prefetcher.next()

            self.save_model_weights(epoch, f"{self.samples_dir}/g_epoch_{epoch + 1}.pth.tar", f"{self.samples_dir}/d_epoch_{epoch + 1}.pth.tar")

    def train_batch(self, real_samples, sd_map, inputs_class_index, target_class_index, fake_noise, rec_noise):
        # 根据正常和缺陷训练方式加载不同类型数据
        real = real_samples.to(self.device, non_blocking=True)
        sd_map = sd_map.to(self.device, non_blocking=True)

        # 接受正常样本，分割图，生成虚假缺陷样本
        real2fake_overlay, real2fake_masks = self.g_model(real, sd_map, fake_noise)
        fake = real * (1 - real2fake_masks) + real2fake_overlay * real2fake_masks
        # 接受虚假缺陷样本，分割图，生成重建正常样本
        fake2real_overlays, fake2real_masks = self.g_model(fake, sd_map, rec_noise)
        rec_real = fake * (1 - fake2real_masks) + fake2real_overlays * fake2real_masks

        # 鉴别器输出虚假缺陷样本的结果
        fake_disc_output, fake_cls_output = self.d_model(fake)

        # 计算图像重建损失
        g_rec_loss = self.rec_criterion(0.5 * (real2fake_masks + fake2real_masks), sd_map[:, [target_class_index.cpu().item()], :, :])
        # 计算鉴别器GP损失
        g_gp_loss = -torch.mean(fake_disc_output)
        # 计算虚假缺陷样本的分类损失
        g_fake_cls_loss = self.cls_criterion(fake_cls_output, target_class_index)
        # 计算真实样本和虚假真实样本的循环一致损失
        g_cycle_rec_loss = self.rec_criterion(real, rec_real)
        # 计算虚假缺陷掩码样本和虚假正常掩码样本的循环一致损失
        g_cycle_mask_rec_loss = self.rec_criterion(real2fake_masks, fake2real_masks)
        # 计算重建掩码损失
        g_cycle_mask_vanishing_loss = -torch.log(torch.mean(
            self.rec_criterion(real2fake_masks, torch.zeros_like(real2fake_masks, device=self.device)) +
            self.rec_criterion(fake2real_masks, torch.zeros_like(fake2real_masks, device=self.device))
        ))
        # 计算重建空间约束损失
        g_cycle_spatial_loss = torch.mean(
            self.rec_criterion(real2fake_masks, torch.zeros_like(real2fake_masks, device=self.device)) +
            self.rec_criterion(fake2real_masks, torch.zeros_like(fake2real_masks, device=self.device))
        )

        # 见论文公式(12)
        g_loss = (
                self.g_gp_loss_weight * g_gp_loss +
                self.g_fake_cls_loss_weight * g_fake_cls_loss +
                self.g_rec_loss_weight * g_rec_loss +
                self.g_cycle_rec_loss_weight * g_cycle_rec_loss +
                self.g_cycle_mask_rec_loss_weight * g_cycle_mask_rec_loss +
                self.g_cycle_mask_vanishing_loss_weight * g_cycle_mask_vanishing_loss +
                self.g_cycle_spatial_loss_weight * g_cycle_spatial_loss
        )

        self.g_optimizer.zero_grad()
        g_loss.backward(retain_graph=True)
        self.g_optimizer.step()

        # 鉴别器输出真实样本的结果
        real_disc_output, real_cls_output = self.d_model(real)

        # 计算鉴别器损失
        d_gp_loss = torch.mean(fake_disc_output) - torch.mean(real_disc_output) + 10 * self.gp_criterion(self.d_model, real, fake)
        d_real_cls_loss = self.cls_criterion(real_cls_output, inputs_class_index)

        d_loss = (
                self.d_gp_loss_weight * d_gp_loss +
                self.d_real_cls_loss_weight * d_real_cls_loss
        )

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        self.g_gp_loss = g_gp_loss
        self.g_fake_cls_loss = g_fake_cls_loss
        self.g_rec_loss = g_rec_loss
        self.g_cycle_rec_loss = g_cycle_rec_loss
        self.g_cycle_mask_rec_loss = g_cycle_mask_rec_loss
        self.g_cycle_mask_vanishing_loss = g_cycle_mask_vanishing_loss
        self.g_cycle_spatial_loss = g_cycle_spatial_loss
        self.d_gp_loss = d_gp_loss
        self.d_real_cls_loss = d_real_cls_loss
        self.g_loss = g_loss
        self.d_loss = d_loss

    def save_model_weights(self, epoch: int, g_model_weights_path: str, d_model_weights_path: str):
        torch.save(
            {
                "start_epoch": epoch + 1,
                "state_dict": self.g_model.state_dict(),
                "ema_state_dict": self.ema_g_model.state_dict() if self.ema_g_model is not None else None,
                "optimizer": self.g_optimizer.state_dict(),
                "scheduler": self.g_scheduler.state_dict() if self.g_scheduler is not None else None,
            },
            g_model_weights_path)
        torch.save(
            {
                "start_epoch": epoch + 1,
                "state_dict": self.d_model.state_dict(),
                "ema_state_dict": self.ema_d_model.state_dict() if self.ema_d_model is not None else None,
                "optimizer": self.d_optimizer.state_dict(),
                "scheduler": self.d_scheduler.state_dict() if self.d_scheduler is not None else None,
            },
            d_model_weights_path)


if __name__ == "__main__":
    # 通过命令行参数读取配置文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/train/defectgan-official.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    trainer = Trainer(config)
    trainer.train()

    # 结束wandb
    wandb.finish()
