# -*- coding: utf-8 -*-
"""
============================================================================
单目深度估计 (Monocular Depth Estimation) - 完整原理解析版
============================================================================

【任务目标】
从单张 RGB 图像估计每个像素的深度值（距离相机的距离）

【核心挑战】
深度估计是一个病态问题（ill-posed problem）：
1. 单张图像丢失了 3D 信息（透视投影）
2. 深度信息具有多尺度特性（需要局部细节 + 全局上下文）
3. 深度边界需要非常精确（物体边缘）

【网络架构】ResNet50 + mHC (Masked Hierarchical Connection) + MaskAttention
├─ Encoder: ResNet50 预训练骨干网络（提取多尺度特征）
├─ Bottleneck: MaskAttention 模块（全局上下文建模）
└─ Decoder: UpProject + mHC 块（特征融合 + 上采样）

【创新点】
1. mHC (Masked Hierarchical Connection): 门控机制自适应融合 Skip Connection
2. MaskAttention: 在 Bottleneck 引入全局注意力
3. Multi-Scale Feature Fusion: 利用 ResNet 的分层特征

过拟合优化版本 - 添加正则化策略
【新增优化】
1. [正则化] 数据增强 (Data Augmentation)
2. [正则化] Dropout 层
3. [正则化] 提前停止 (Early Stopping)
4. [正则化] 增大验证集比例
5. [正则化] 标签平滑 (Label Smoothing for Depth)
6. [正则化] 梯度裁剪 (Gradient Clipping)
============================================================================

"""

import os
import sys
import time
import subprocess
import requests
import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import pipeline

try:
    from torch.amp import autocast, GradScaler
    AUTOCAST_DEVICE = 'cuda'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AUTOCAST_DEVICE = None

# -----------------------------------------------------------------------------
# 0. 全局常量配置
# -----------------------------------------------------------------------------
MAX_DEPTH = 10.0
INPUT_HEIGHT = 480
INPUT_WIDTH = 640
TRAIN_RATIO = 0.8  #  从 0.98 改为 0.8，增大验证集
NUM_WORKERS = 4
NUM_EPOCHS = 20  #  增加训练轮数，配合 Early Stopping
BATCH_SIZE_PER_GPU = 10  #  从 8 减小到 6，更稳定
SEED = 40

USE_AMP = True
USE_DATA_AUGMENTATION = True  #  数据增强开关
USE_DROPOUT = True  #  Dropout 开关
USE_EARLY_STOPPING = True  #  Early Stopping 开关
EARLY_STOP_PATIENCE = 25  #  15 个 epoch 不提升就停止
GRAD_CLIP_VALUE = 1.0  #  梯度裁剪阈值

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)  #  也固定 random 模块

set_seed(SEED)

# -----------------------------------------------------------------------------
# 1. 环境配置
# -----------------------------------------------------------------------------
def install_dependencies():
    print("[环境准备] 正在检查并安装必要依赖...")
    packages = ["transformers", "h5py", "timm", "accelerate"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    print("[环境准备] 完成。")

install_dependencies()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前计算设备: {device}")
if torch.cuda.is_available():
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"混合精度: {'启用' if USE_AMP else '禁用'}")
    print(f"数据增强: {'启用' if USE_DATA_AUGMENTATION else '禁用'}")
    print(f"Dropout: {'启用' if USE_DROPOUT else '禁用'}")
    print(f"Early Stopping: {'启用 (patience='+str(EARLY_STOP_PATIENCE) if USE_EARLY_STOPPING else '禁用'}")

def get_autocast(enabled=True):
    if AUTOCAST_DEVICE is not None:
        return autocast(device_type=AUTOCAST_DEVICE, enabled=enabled)
    else:
        return autocast(enabled=enabled)

# -----------------------------------------------------------------------------
# 2. 数据增强策略
# -----------------------------------------------------------------------------
class MultiScaleRandomCrop:
    """随机选择不同尺度裁剪 - 确保 RGB 和 Depth 同步"""
    def __init__(self, scales=[(320, 240), (384, 288), (480, 360)], p=0.2):
        self.scales = scales
        self.p = p

    def __call__(self, img, depth):
        # 随机选择一个尺度
        if random.random() < self.p:
            size = random.choice(self.scales)
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=size)
            img = transforms.functional.crop(img, i, j, h, w)
            depth = transforms.functional.crop(depth, i, j, h, w)

        return img, depth

class DepthAwareRandomHorizontalFlip:
    """深度图感知的随机水平翻转"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, depth):
        if random.random() < self.p:
            img = transforms.functional.hflip(img)
            depth = transforms.functional.hflip(depth)
        return img, depth

class DepthAwareColorJitter:
    """只对 RGB 图像做颜色扰动，不影响深度图"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        self.p = p
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, depth):
        if random.random() < self.p:
            img = self.color_jitter(img)
        return img, depth

# -----------------------------------------------------------------------------
# 3. 数据集加载
# -----------------------------------------------------------------------------
class DataDownloader:
    def __init__(self, dest_dir="data"):
        self.dest_dir = dest_dir
        os.makedirs(self.dest_dir, exist_ok=True)
        self.url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
        self.filepath = os.path.join(self.dest_dir, "nyu_depth_v2_labeled.mat")

    def download(self):
        if os.path.exists(self.filepath):
            return self.filepath
        print(f"[下载中] 正在下载数据集 (约 2.8 GB)...")
        try:
            with requests.get(self.url, stream=True) as r:
                r.raise_for_status()
                with open(self.filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192*8):
                        if chunk: f.write(chunk)
            print("[完成] 下载成功。")
            return self.filepath
        except Exception as e:
            print(f"[错误] 下载失败: {e}")
            return None

class NYUDataset(Dataset):
    def __init__(self, mat_file_path, split='train', train_ratio=TRAIN_RATIO,
                 transform=None, target_transform=None, use_augmentation=False):
        self.file_path = mat_file_path
        self.transform = transform
        self.target_transform = target_transform
        self.use_augmentation = use_augmentation  #  增强开关

        if not os.path.exists(mat_file_path):
            raise FileNotFoundError(f"找不到文件: {mat_file_path}")

        print(f"[{split.upper()}] 正在将数据加载到内存中...")

        with h5py.File(self.file_path, 'r') as f:
            total_samples = f['images'].shape[0]
            indices = np.arange(total_samples)
            np.random.shuffle(indices)

            split_idx = int(total_samples * train_ratio)
            if split == 'train':
                self.indices = indices[:split_idx]
            else:
                self.indices = indices[split_idx:]

            raw_images = f['images'][:]
            raw_depths = f['depths'][:]

        self.images = raw_images[self.indices]
        self.depths = raw_depths[self.indices]

        del raw_images
        del raw_depths

        print(f"[{split.upper()}] 加载完成，样本数: {len(self.images)}")

        #  数据增强 pipeline (只在训练时使用)
        if self.use_augmentation:
            self.augmentations = [
                MultiScaleRandomCrop(scales=[(320, 240), (384, 288), (480, 360)]),
                DepthAwareRandomHorizontalFlip(p=0.5),
                DepthAwareColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        else:
            self.augmentations = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np = self.images[idx]
        depth_np = self.depths[idx]

        img_np = np.transpose(img_np, (2, 1, 0))
        depth_np = np.transpose(depth_np, (1, 0))

        image = Image.fromarray(np.uint8(img_np))
        depth = Image.fromarray(depth_np)

        #  应用数据增强
        for aug in self.augmentations:
            image, depth = aug(image, depth)

        if self.transform: image = self.transform(image)
        if self.target_transform: depth = self.target_transform(depth)

        return image, depth

# -----------------------------------------------------------------------------
# 4. 添加 Dropout 的模型架构
# -----------------------------------------------------------------------------
class MaskAttentionModule(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.0):  #  添加 dropout 参数
        super(MaskAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)

        #  添加 Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),  #  FFN 中也加 Dropout
            nn.Conv2d(channels * 2, channels, kernel_size=1)
        )

        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        shortcut = x

        q = self.q_proj(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = self.k_proj(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = self.v_proj(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        #  应用 Dropout
        attn_out = self.dropout(attn_out)

        x = self.norm1(shortcut + self.gamma * attn_out)
        x = self.norm2(x + self.ffn(x))

        return x

class MHCBlock(nn.Module):
    def __init__(self, skip_channels, decoder_channels, gate_ratio=1.0):
        super(MHCBlock, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(skip_channels, decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )

        gate_hidden_dim = int(decoder_channels * gate_ratio)

        self.gate = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, gate_hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden_dim, decoder_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, skip, x):
        skip_proj = self.proj(skip)
        combined = torch.cat([skip_proj, x], dim=1)
        mask = self.gate(combined)
        skip_refined = skip_proj * mask
        return torch.cat([x, skip_refined], dim=1)

class UpProjectMHC(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):  #  添加 dropout
        super(UpProjectMHC, self).__init__()

        self.mhc = MHCBlock(skip_channels, in_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),  # ✨ Dropout
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_feature):
        target_size = skip_feature.shape[2:]
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)

        x_combined = self.mhc(skip_feature, x)
        return self.conv(x_combined)

class DepthNetResNet50MHC(nn.Module):
    def __init__(self, use_dropout=USE_DROPOUT):
        super(DepthNetResNet50MHC, self).__init__()
        print("初始化 ResNet-50 + mHC 模型...")
        original_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.encoder_conv1 = nn.Sequential(original_resnet.conv1, original_resnet.bn1, original_resnet.relu)
        self.encoder_maxpool = original_resnet.maxpool
        self.encoder_layer1 = original_resnet.layer1
        self.encoder_layer2 = original_resnet.layer2
        self.encoder_layer3 = original_resnet.layer3
        self.encoder_layer4 = original_resnet.layer4

        #  根据配置决定 Dropout 比例
        dropout_rate = 0.1 if use_dropout else 0.0

        self.bottleneck_attn = MaskAttentionModule(channels=2048, dropout=dropout_rate)

        self.up1 = UpProjectMHC(2048, 1024, 1024, dropout=dropout_rate)
        self.up2 = UpProjectMHC(1024, 512, 512, dropout=dropout_rate)
        self.up3 = UpProjectMHC(512, 256, 256, dropout=dropout_rate)
        self.up4 = UpProjectMHC(256, 64, 64, dropout=dropout_rate * 0.5)  # 最后一层少一点

        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.encoder_conv1(x)
        x_pool = self.encoder_maxpool(x0)
        x1 = self.encoder_layer1(x_pool)
        x2 = self.encoder_layer2(x1)
        x3 = self.encoder_layer3(x2)
        x4 = self.encoder_layer4(x3)

        x4 = self.bottleneck_attn(x4)

        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        d4 = self.up4(d3, x0)

        out = self.final_conv(d4)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return self.sigmoid(out)

# -----------------------------------------------------------------------------
# 5. Loss 与 评价指标
# -----------------------------------------------------------------------------
class SILogLoss(nn.Module):
    def __init__(self, lamb=0.5, max_depth=MAX_DEPTH):
        super(SILogLoss, self).__init__()
        self.lamb = lamb
        self.max_depth = max_depth

    def forward(self, pred, target):
        pred = pred * self.max_depth
        valid_mask = target > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_val = pred[valid_mask]
        target_val = target[valid_mask]

        pred_val = torch.clamp(pred_val, min=1e-3)
        target_val = torch.clamp(target_val, min=1e-3)

        log_diff = torch.log(pred_val) - torch.log(target_val)
        mse_log = torch.mean(log_diff ** 2)
        mean_log_sq = torch.mean(log_diff) ** 2

        return torch.sqrt(mse_log - self.lamb * mean_log_sq)

def compute_depth_metrics(pred, target, max_depth=MAX_DEPTH):
    pred = pred * max_depth
    valid_mask = target > 0

    pred = pred[valid_mask]
    target = target[valid_mask]

    pred = torch.clamp(pred, min=1e-3)
    target = torch.clamp(target, min=1e-3)

    thresh = torch.max((target / pred), (pred / target))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = torch.sqrt(torch.mean((target - pred) ** 2))

    return {'a1': a1.item(), 'a2': a2.item(), 'a3': a3.item(), 'rmse': rmse.item()}

# -----------------------------------------------------------------------------
# 6. Early Stopping 类
# -----------------------------------------------------------------------------
class EarlyStopping:
    """Early Stopping 防止过拟合"""
    def __init__(self, patience=10, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  [Early Stopping] 验证损失未改善: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  [Early Stopping] 触发！停止训练。")
        else:
            self.best_loss = val_loss
            self.counter = 0

# -----------------------------------------------------------------------------
# 7. 训练与验证
# -----------------------------------------------------------------------------
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metrics = {'a1': 0, 'a2': 0, 'a3': 0, 'rmse': 0}
    num_batches = len(val_loader)

    with torch.no_grad():
        for imgs, depths in val_loader:
            imgs, depths = imgs.to(device), depths.to(device)

            if USE_AMP:
                with get_autocast(enabled=True):
                    preds = model(imgs)
                    loss = criterion(preds, depths)
            else:
                preds = model(imgs)
                loss = criterion(preds, depths)

            total_loss += loss.item()
            batch_metrics = compute_depth_metrics(preds, depths)
            for k, v in batch_metrics.items():
                metrics[k] += v

    avg_loss = total_loss / num_batches
    for k in metrics: metrics[k] /= num_batches
    return avg_loss, metrics

def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_path='best_model.pth'):
    if not os.path.exists(checkpoint_path):
        print(f"[Init] 未发现 Checkpoint，将开始全新训练。")
        return False, 0, float('inf')

    print(f"[Init] 发现 Checkpoint，正在加载...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, OrderedDict) or 'model_state_dict' not in checkpoint:
            state_dict = checkpoint
            epoch = 0
            best_val_loss = float('inf')
        else:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(new_state_dict)

        print(f"[Init] 加载成功！Epoch {epoch}, Best Loss: {best_val_loss:.4f}")
        return True, epoch, best_val_loss

    except Exception as e:
        print(f"[Init] 加载失败: {e}")
        return False, 0, float('inf')

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, metrics, checkpoint_path='best_model.pth'):
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': val_loss,
        'metrics': metrics
    }

    torch.save(checkpoint, checkpoint_path)

def main():
    downloader = DataDownloader()
    mat_file_path = downloader.download()
    if not mat_file_path: return

    #  数据预处理（验证集不做增强）
    data_transforms = transforms.Compose([
        transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    depth_transforms = transforms.Compose([
        transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
        transforms.ToTensor()
    ])

    #  训练集启用数据增强
    train_dataset = NYUDataset(
        mat_file_path, 'train', TRAIN_RATIO,
        data_transforms, depth_transforms,
        use_augmentation=USE_DATA_AUGMENTATION
    )
    val_dataset = NYUDataset(
        mat_file_path, 'val', TRAIN_RATIO,
        data_transforms, depth_transforms,
        use_augmentation=False  # 验证集不增强
    )

    gpu_count = torch.cuda.device_count()
    BATCH_SIZE = BATCH_SIZE_PER_GPU * max(1, gpu_count)
    print(f"Batch Size: {BATCH_SIZE}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    scaler = GradScaler(enabled=USE_AMP)
    model = DepthNetResNet50MHC(use_dropout=USE_DROPOUT)

    if gpu_count > 1:
        print(f"启用 {gpu_count} 个 GPU 进行训练...")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = SILogLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  #  增大 weight_decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    #  Early Stopping
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, verbose=True) if USE_EARLY_STOPPING else None

    has_checkpoint, start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, 'best_model.pth')

    print(f"\n>>> 开始训练 (最多 {NUM_EPOCHS} Epochs)...")
    print(f"正则化策略:")
    print(f"  - 数据增强: {'✓' if USE_DATA_AUGMENTATION else '✗'}")
    print(f"  - Dropout (0.1): {'✓' if USE_DROPOUT else '✗'}")
    print(f"  - Early Stopping: {'✓' if USE_EARLY_STOPPING else '✗'}")
    print(f"  - 梯度裁剪: ✓ (max_norm={GRAD_CLIP_VALUE})")
    print(f"  - Weight Decay: 1e-4")
    print(f"  - 验证集比例: {1-TRAIN_RATIO:.1%}\n")

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
            running_loss = 0.0

            for imgs, depths in loop:
                imgs, depths = imgs.to(device), depths.to(device)

                optimizer.zero_grad()

                with get_autocast(enabled=USE_AMP):
                    preds = model(imgs)
                    loss = criterion(preds, depths)

                scaler.scale(loss).backward()

                #  梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)

                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            scheduler.step()

            val_loss, val_metrics = validate(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {running_loss / len(train_loader):.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Metrics:    a1={val_metrics['a1']:.3f}, a2={val_metrics['a2']:.3f}, RMSE={val_metrics['rmse']:.3f}")
            print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_metrics, 'best_model.pth')
                print("  [*] 新记录！模型已保存")

            #  Early Stopping 检查
            if early_stopping:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f"\n[Early Stopping] 在 Epoch {epoch+1} 停止训练")
                    break

    except KeyboardInterrupt:
        print("\n训练被手动中断。")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()

    # 可视化
    print("\n>>> 可视化最佳模型结果...")
    if os.path.exists('best_model.pth'):
        load_checkpoint(model, checkpoint_path='best_model.pth')

    model.eval()
    sample_img, sample_depth = val_dataset[random.randint(0, len(val_dataset)-1)]
    with torch.no_grad():
        input_tensor = sample_img.unsqueeze(0).to(device)
        pred_depth = model(input_tensor) * MAX_DEPTH
        pred_depth = pred_depth.squeeze().cpu().numpy()

    img_np = sample_img.permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(img_np); plt.title("Input RGB")
    plt.subplot(1, 3, 2); plt.imshow(sample_depth.squeeze(), cmap='magma_r'); plt.title("Ground Truth")
    plt.subplot(1, 3, 3); plt.imshow(pred_depth, cmap='magma_r'); plt.title(f"Prediction Loss: {best_val_loss:.3f}")
    plt.savefig(f'{best_val_loss:.3f}result_vis.png', dpi=200)
    plt.show()

    print("加载 Depth Anything V2 Small 模型...")
    # 该管道会自动下载模型权重 (~100MB)
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=0 if torch.cuda.is_available() else -1)

    # 使用验证集中的一张图片进行测试
    # sample_img_tensor, _ = val_dataset[random.randint(0, len(val_dataset) - 1)]
    # sample_img, sample_depth = val_dataset[random.randint(0, len(val_dataset) - 1)]

    # 需要将 Tensor 转回 PIL Image 给 pipeline 使用
    # 先反归一化
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    sample_img_pil = transforms.ToPILImage()(inv_normalize(sample_img))

    # 推理
    start_t = time.time()
    result = pipe(sample_img_pil)
    print(f"推理耗时: {time.time() - start_t:.4f} 秒")

    # 结果包含 'depth' (PIL Image) 和 'predicted_depth' (Tensor)
    pred_depth_img = result["depth"]

    # 可视化对比
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_img_pil, cmap='magma_r')
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_depth_img)
    plt.title("Depth Anything V2 Prediction")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
"""
============================================================================
【总结】

这个深度估计网络的核心创新和设计思想：

1. **Encoder-Decoder 架构**
   - Encoder: ResNet50 提取多尺度特征
   - Decoder: 逐步上采样恢复分辨率
   - Skip Connection: 融合低层细节

2. **mHC (Masked Hierarchical Connection)**
   - 问题：传统 Skip Connection 盲目拼接
   - 方案：门控机制自适应选择有用特征
   - 优势：减少语义鸿沟，提升精度

3. **MaskAttention**
   - 问题：深度估计需要全局上下文
   - 方案：Self-Attention 捕获长距离依赖
   - 位置：Bottleneck（最深层）

4. **SILog Loss**
   - 问题：MSE 对尺度敏感，被大值主导
   - 方案：log 空间的相对误差
   - 优势：尺度不变，关注相对关系

5. **正则化策略**
   - 数据增强：扩充训练样本
   - Dropout: 防止共适应
   - Early Stopping: 防止过拟合
   - Weight Decay: L2 正则化
   - 梯度裁剪: 训练稳定

6. **工程优化**
   - 混合精度训练: 加速 1.5-2x
   - 数据预加载: 避免 I/O 瓶颈
   - 多 GPU 并行: 扩展吞吐量

【适用场景】
- 室内深度估计（NYU Depth V2）
- 自动驾驶（KITTI）
- 机器人导航
- AR/VR 应用

【改进方向】
1. 更先进的 Encoder: Swin Transformer, ConvNeXt
2. 更好的融合模块: FPN, BiFPN
3. 多任务学习: 深度 + 法向量 + 语义分割
4. 自监督学习: 利用未标注视频数据

【参考文献】
- [1] Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network", NeurIPS 2014
- [2] Laina et al., "Deeper Depth Prediction with Fully Convolutional Residual Networks", 3DV 2016
- [3] Fu et al., "Deep Ordinal Regression Network for Monocular Depth Estimation", CVPR 2018
- [4] Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer", TPAMI 2020
日期：2025-02-13
