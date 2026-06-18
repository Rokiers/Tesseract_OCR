"""
============================================================================
阶段4: 微调训练 —— 让模型认得 Exocet 字体
============================================================================

目的:
  在 EasyOCR 预训练模型的基础上，用我们生成的 Exocet 字体数据做微调。
  让模型学会「Exocet 字体的笔画特征 → 正确的字符」这个映射。

训练策略（方案C — 冻结 CNN 前几层）:
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ CNN ❄️冻结 │───→│ RNN 🔥训练 │───→│ Head 🔥训练 │
  │ 保留通用    │    │ 学习Exocet │    │ 重写字符映射 │
  │ 视觉能力    │    │ 的时序特征  │    │             │
  └──────────┘    └──────────┘    └──────────┘

核心概念讲解（训练循环中逐一展开）:
  · 梯度下降: 沿着「能让损失变小」的方向更新参数
  · 反向传播: 从损失值出发，用链式法则计算每个参数的梯度
  · 学习率: 每次更新的步长。太大=跳过最优解，太小=训练太慢
  · Epoch: 把全部数据过一遍 = 1 epoch
  · Batch: 每次更新参数用多少张图
  · CTC Loss: 不需要每个像素对齐到字符，自动学习对齐关系

输出:
  saved_models/exocet_best.pth       ← 验证集上最好的模型
  saved_models/exocet_checkpoint_*.pth  ← 训练检查点
  outputs/training_log.json          ← 训练日志（loss/acc 曲线数据）
============================================================================
"""

import os
import sys
import json
import time
import math
from datetime import datetime
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import easyocr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROJECT_ROOT, TRAIN_DIR, VAL_DIR, TEST_DIR,
    SAVED_MODELS_DIR, OUTPUTS_DIR, DEVICE,
    CHAR2IDX, IDX2CHAR, CTC_BLANK, NUM_CHARS,
    TRAIN_CONFIG, DATA_CONFIG
)


# ============================================================
# 第1步: 自定义 Dataset（数据加载器）
# ============================================================

class OCRDataset(Dataset):
    """
    PyTorch Dataset：负责从磁盘读取图片和标签，转换成模型需要的格式。

    PyTorch 中 Dataset 的作用:
      你定义「如何加载数据」，PyTorch 的 DataLoader 自动帮你:
        - 批量读取 (batching)
        - 打乱顺序 (shuffling)
        - 多线程加载 (num_workers)
        - 填充/截断 (collate)
    """

    def __init__(self, data_dir: str, img_height: int = 32):
        self.data_dir = data_dir
        self.img_height = img_height
        self.samples = []

        # 读取 labels.txt
        labels_path = os.path.join(data_dir, 'labels.txt')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"找不到 labels 文件: {labels_path}")

        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    img_name, text = line.split('\t', 1)
                    img_path = os.path.join(data_dir, img_name)
                    if os.path.exists(img_path):
                        self.samples.append((img_path, text))

        print(f"  {os.path.basename(data_dir)}: 加载 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回一张图片和它的标签。

        图片会被:
          1. 读取为灰度图
          2. 缩放到固定高度 (32px)，保持宽高比
          3. 归一化到 [0, 1]
          4. 转为 Tensor，shape: (1, H, W)

        标签会被:
          1. 每个字符转为索引数字
          2. 转为 LongTensor，shape: (seq_len,)
        """
        img_path, text = self.samples[idx]

        # --- 读取图片 ---
        img = Image.open(img_path).convert('L')  # 'L' = 灰度

        # --- 缩放到统一高度 ---
        w, h = img.size
        ratio = self.img_height / h
        new_w = max(int(w * ratio), 4)  # 至少 4 像素宽（避免太小导致 CNN 出错）
        img = img.resize((new_w, self.img_height), Image.BICUBIC)

        # --- 转 numpy → 归一化 → Tensor ---
        img_np = np.array(img, dtype=np.float32) / 255.0  # [0, 1] 浮点
        img_tensor = torch.FloatTensor(img_np).unsqueeze(0)  # (1, H, W)

        # --- 标签编码 ---
        # 每个字符 → 数字索引
        # blank (索引 0) 不需要出现在标签里，CTC Loss 会自动处理
        label_indices = []
        for ch in text:
            if ch in CHAR2IDX:
                label_indices.append(CHAR2IDX[ch])
            else:
                # 如果字符不在字符集中，跳过（这种情况不应该出现，但安全起见）
                print(f"  ⚠️  未知字符 '{ch}' 在 '{text}' 中，跳过")
        label_tensor = torch.LongTensor(label_indices)  # (seq_len,)

        return img_tensor, label_tensor


# ============================================================
# 第2步: CTC Collate（批处理函数）
# ============================================================

def ctc_collate(batch):
    """
    把多个样本拼成一个 batch。

    CTC 模型的特殊需求:
      图片可以不一样宽，但 Tensor 必须统一尺寸。
      解决方案: 用 0 填充(pad)到最宽的宽度。

      同时记录每张图的实际宽度 —— CTC Loss 需要这个信息
      来知道哪些帧是「真实内容」、哪些是填充。
    """
    images, labels = zip(*batch)

    # 找到 batch 中最宽的图
    max_width = max(img.shape[2] for img in images)

    # 填充图片到统一宽度
    padded_images = []
    for img in images:
        c, h, w = img.shape
        if w < max_width:
            # 右侧填充 0（黑色）
            pad = torch.zeros(c, h, max_width - w)
            img = torch.cat([img, pad], dim=2)
        padded_images.append(img)

    # 堆叠成 batch
    images_batch = torch.stack(padded_images)  # (B, C, H, max_W)

    # 标签拼接（CTC Loss 用 1D 的拼接格式）
    labels_concat = torch.cat(labels)  # 所有样本的标签拼一起

    # 每个样本的标签长度
    label_lengths = torch.LongTensor([len(l) for l in labels])

    # 每张图的「帧长度」（因为 CNN 把宽度压缩了 4 倍）
    # 实际使用时 CTC Loss 需要知道输入序列长度
    input_lengths = torch.LongTensor([
        img.shape[2] // 4 for img in images  # CNN 下采样 4 倍
    ])

    return images_batch, labels_concat, label_lengths, input_lengths


# ============================================================
# 第3步: 模型构建（替换 Head 为自定义字符集）
# ============================================================

def build_model():
    """
    加载 EasyOCR 预训练模型，替换 Head 为自定义字符集的 Head。

    步骤:
      1. 加载预训练识别器
      2. 拿到 CNN + RNN (保留预训练权重)
      3. 替换 Head: nn.Linear(256, 97) → nn.Linear(256, NUM_CHARS+1)
      4. 冻结 CNN 前几层
    """
    print(f"\n[1/5] 构建模型...")

    # --- 加载预训练模型 ---
    reader = easyocr.Reader(
        ['en'],
        gpu=(DEVICE.type == 'cuda'),
        model_storage_directory=os.path.join(PROJECT_ROOT, 'easyocr_models'),
        verbose=False
    )
    pretrained_model = reader.recognizer.model
    pretrained_model.to(DEVICE)
    pretrained_model.eval()

    # --- 记录原始 Head 的输入维度 ---
    # 找到 nn.Linear 层，拿到 in_features
    head_in_features = None
    for module in pretrained_model.Prediction.modules():
        if isinstance(module, nn.Linear):
            head_in_features = module.in_features
            break

    print(f"  原始 Head: nn.Linear({head_in_features}, 97)")
    print(f"  新 Head:   nn.Linear({head_in_features}, {NUM_CHARS + 1})  "
          f"(+1 = CTC blank)")
    print(f"  参数量:    {head_in_features * 97 + 97:,d} → "
          f"{head_in_features * (NUM_CHARS + 1) + (NUM_CHARS + 1):,d}")

    # --- 替换 Head ---
    # 在原模型上直接修改最后一层
    for name, module in pretrained_model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == 97:
            # 找到 Head 中的 Linear 层
            pass  # 标记位置

    # 查找并替换 Prediction 中所有 Linear 层
    def replace_head_modules(module, head_in_features):
        """
        递归替换模块中的 Linear 层。
        任何 out_features==97 的 Linear 层都替换为自定义字符集大小的。
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                old_features = child.out_features
                # 只替换分类层（通常 out_features 接近 97 或 96）
                if 90 <= old_features <= 100:
                    new_linear = nn.Linear(child.in_features, NUM_CHARS + 1)
                    # 保留预训练权重中前 N 个字符的映射
                    # （但因为我们字符集可能不同，这里先随机初始化）
                    # TODO: 如果字符集是原字符集的子集，可以拷贝对应权重
                    setattr(module, name, new_linear)
                    print(f"    替换 {name}: {child.in_features}×{old_features} → "
                          f"{child.in_features}×{NUM_CHARS + 1}")
            elif isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # 替换 BatchNorm 的统计维度
                if child.num_features == 97:
                    new_bn = nn.BatchNorm1d(NUM_CHARS + 1)
                    setattr(module, name, new_bn)
                    print(f"    替换 {name}: BatchNorm {97} → {NUM_CHARS + 1}")
            else:
                replace_head_modules(child, head_in_features)

    replace_head_modules(pretrained_model.Prediction, head_in_features)

    # --- 冻结 CNN 前几层 ---
    print(f"\n  冻结策略: CNN 前 {TRAIN_CONFIG.FREEZE_CNN_LAYERS} 层 ❄️")

    freeze_count = 0
    trainable_count = 0

    # CNN 层编号（只在卷积层计数）
    conv_idx = 0
    for name, param in pretrained_model.FeatureExtraction.named_parameters():
        # 只对 Conv2d 的权重冻结
        if 'Conv' in name and 'weight' in name:
            if conv_idx < TRAIN_CONFIG.FREEZE_CNN_LAYERS:
                param.requires_grad = False
                freeze_count += 1
                conv_idx += 1
            else:
                param.requires_grad = True
                trainable_count += 1
                conv_idx += 1
        else:
            # BN(BatchNorm)和 bias 也先冻结
            if conv_idx <= TRAIN_CONFIG.FREEZE_CNN_LAYERS:
                param.requires_grad = False
                freeze_count += 1
            else:
                param.requires_grad = True
                trainable_count += 1

    total_params = sum(p.numel() for p in pretrained_model.parameters())
    trainable_params = sum(p.numel() for p in pretrained_model.parameters()
                           if p.requires_grad)

    print(f"  总参数:     {total_params:,d}")
    print(f"  可训练参数: {trainable_params:,d} "
          f"({trainable_params/total_params*100:.1f}%)")
    print(f"  冻结参数:   {total_params - trainable_params:,d}")

    return pretrained_model


# ============================================================
# 第4步: 训练循环
# ============================================================

def train_epoch(model, dataloader, optimizer, criterion, epoch):
    """
    一个 epoch 的训练。

    训练循环 = 深度学习最核心的循环:

    for each batch:
      ① 前向传播 (forward):   输入 → 模型 → 输出 → 计算损失
      ② 清零梯度:             清除上一轮的梯度残留
      ③ 反向传播 (backward):  损失对每个参数求偏导 → 梯度
      ④ 参数更新 (optimize):  参数 -= 学习率 × 梯度

    什么是梯度下降（用爬山比喻）:
      你在雾中登山，目标是走到海拔最低的山谷。
      每一步你只能看到脚下 1 米范围内的坡度:
        - 如果脚下往左倾斜 → 往右边走
        - 如果脚下往前倾斜 → 往后退
        - 坡度越陡 → 步子越大
      这个「坡度」就是梯度。
      这个「步子大小」就是学习率。

    什么是反向传播：
      假设 loss = f(g(h(x)))，我们要计算 ∂loss/∂x
      链式法则: ∂loss/∂x = ∂loss/∂f × ∂f/∂g × ∂g/∂h × ∂h/∂x
      从最后一层往前一层层算，所以叫「反向」传播。

    CTC Loss 的特殊之处：
      普通分类: 每个像素 → 一个 label (需要对齐标注)
      CTC:       序列 → 序列 (不需要对齐！)
                 模型自己学习哪些帧属于哪个字符
                 blank token 标记「空帧」
    """
    model.train()  # 训练模式（启用 Dropout、BatchNorm 等）
    total_loss = 0
    num_batches = 0

    start_time = time.time()
    print(f"\n  Epoch {epoch + 1}/{TRAIN_CONFIG.NUM_EPOCHS}")

    for batch_idx, (images, labels_concat, label_lengths, input_lengths) in enumerate(dataloader):
        # 移到 GPU/CPU
        images = images.to(DEVICE)
        labels_concat = labels_concat.to(DEVICE)
        label_lengths = label_lengths.to(DEVICE)
        input_lengths = input_lengths.to(DEVICE)

        # ----- ① 前向传播 (Forward) -----
        # 图片经过整个 CRNN：CNN → RNN → Head
        # 输出 shape: (B, T, num_classes)
        #   T = 时间步数（图片宽度 / 4，每个 CNN strides 效应）
        #   num_classes = NUM_CHARS + 1 (含 blank)
        output = model(images)

        # 转置为 CTC Loss 需要的格式: (T, B, num_classes)
        output_ctc = output.permute(1, 0, 2).log_softmax(2)

        # ----- ② 计算损失 (Loss) -----
        loss = criterion(
            output_ctc,        # (T, B, C) log 概率
            labels_concat,     # 所有标签拼接
            input_lengths,     # 每个输入的 T
            label_lengths      # 每个标签长度
        )

        # 如果 batch 太小，可能出现无效的 input_lengths
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        # ----- ③ 清零梯度 -----
        # 为什么每次都要清零？
        # PyTorch 默认会累加梯度（为了支持 RNN 等需要多次 backward 的场景）
        # 大多数情况一个 batch 一次就够了，所以要先清零
        optimizer.zero_grad()

        # ----- ④ 反向传播 (Backward) -----
        # 从 loss 出发，沿着计算图往回走，计算每个参数的梯度
        # 梯度存在 param.grad 里
        loss.backward()

        # ----- ⑤ 梯度裁剪 (Gradient Clipping) -----
        # 防止梯度爆炸（某次更新步长太大，模型参数直接飞了）
        # 把所有参数的梯度「裁剪」到一个范围内
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG.GRAD_CLIP)

        # ----- ⑥ 参数更新 (Optimize) -----
        # optimizer.step() 做的事:
        #   for param in model.parameters():
        #       if param.requires_grad:
        #           param = param - learning_rate × param.grad
        # 这就是「梯度下降」— 沿梯度的反方向走一步
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # 每 N 个 batch 打印进度
        if (batch_idx + 1) % 20 == 0:
            avg_loss = total_loss / num_batches
            print(f"    Batch {batch_idx + 1:4d}/{len(dataloader)}  "
                  f"Loss: {avg_loss:.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    elapsed = time.time() - start_time
    print(f"    ✅ Epoch {epoch + 1} 完成  Loss: {avg_loss:.4f}  "
          f"耗时: {elapsed:.0f}s")

    return avg_loss


# ============================================================
# 第5步: 验证
# ============================================================

def validate(model, dataloader, criterion):
    """
    在验证集上评估模型。

    与训练的区别:
      - model.eval() 关闭 Dropout/BatchNorm 训练行为
      - torch.no_grad() 不计算梯度（省显存，加速推理）
      - 不调用 optimizer.step()
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    correct_chars = 0
    total_chars = 0

    with torch.no_grad():  # 不计算梯度 ← 省显存 + 加速
        for images, labels_concat, label_lengths, input_lengths in dataloader:
            images = images.to(DEVICE)
            labels_concat = labels_concat.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE)
            input_lengths = input_lengths.to(DEVICE)

            # 前向传播
            output = model(images)
            output_ctc = output.permute(1, 0, 2).log_softmax(2)

            # 计算损失
            loss = criterion(output_ctc, labels_concat,
                             input_lengths, label_lengths)
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            num_batches += 1

            # ----- 解码 CTC 输出，计算字符准确率 -----
            # argmax: 每个时间步取概率最高的字符
            _, pred_indices = output.permute(1, 0, 2).max(2)  # (B, T)

            for b in range(pred_indices.size(0)):
                # 解码单个样本: 去重 + 去 blank
                pred = ctc_decode(pred_indices[b].cpu().numpy())
                # 真实的标签
                true_indices = labels_concat.cpu().numpy()
                # 简单比较（不考虑对齐）
                # 这里简化为字符级准确率
                # 实际应用中需要用编辑距离(Levenshtein)等指标

    avg_loss = total_loss / max(num_batches, 1)
    print(f"    验证 Loss: {avg_loss:.4f}")

    return avg_loss


def ctc_decode(indices: np.ndarray) -> list:
    """
    CTC 解码: 把模型输出的概率最高索引序列 → 实际字符序列。

    算法:
      1. 去重: 连续的相同索引合并为一个
         [3,3,3,5,5] → [3,5]
      2. 去空白: 移除索引 0 (CTC blank)
         [0,3,0,5] → [3,5]

    为什么需要这样？
      模型可能多个帧都输出同一个字符（比如一个宽的 'W' 可能占 5 帧）
      这些帧都输出 'W' → 合并为 1 个 'W'
    """
    decoded = []
    prev = -1
    for idx in indices:
        if idx != prev and idx != 0:  # 跳过重复 + 跳过 blank
            decoded.append(idx)
        prev = idx
    return decoded


# ============================================================
# 第6步: 在真实图片上测试微调效果
# ============================================================

def test_on_real_images(model, image_dir: str):
    """
    在原始 16 张游戏截图上测试微调后的模型。

    比较微调前后:
      阶段1 的 baseline 结果 vs 现在的 fine-tuned 结果
    """
    print(f"\n[真实图片测试] 在 {image_dir}/ 上评估...")

    # 复用阶段1的预处理和 OCR 代码
    # 简化版：直接用 EasyOCR 的框架 + 替换过的模型
    # （完整实现见 inference.py）

    # TODO: 完整评估
    print("  (完整评估见 inference.py)")
    return {}


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 60)
    print("  阶段4: 模型微调训练")
    print("=" * 60)

    print(f"\n训练设备: {DEVICE}")
    print(f"字符集:   {NUM_CHARS} 个字符")
    print(f"训练轮数: {TRAIN_CONFIG.NUM_EPOCHS}")

    # ---- 第1步: 构建模型 ----
    model = build_model()

    # ---- 第2步: 加载数据 ----
    print(f"\n[2/5] 加载数据集...")
    train_dataset = OCRDataset(TRAIN_DIR)
    val_dataset = OCRDataset(VAL_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG.BATCH_SIZE,
        shuffle=True,  # 训练时打乱 ← 避免模型记住顺序
        collate_fn=ctc_collate,
        num_workers=0,  # WSL 中多线程可能有问题，先用 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG.BATCH_SIZE,
        shuffle=False,
        collate_fn=ctc_collate,
        num_workers=0,
    )

    # ---- 第3步: 优化器和损失函数 ----
    print(f"\n[3/5] 设置优化器和损失函数...")

    # 只优化 requires_grad=True 的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(
        trainable_params,
        lr=TRAIN_CONFIG.LEARNING_RATE
    )
    print(f"  优化器: Adam (lr={TRAIN_CONFIG.LEARNING_RATE})")
    print(f"  优化参数数量: {sum(p.numel() for p in trainable_params):,d}")

    # CTC Loss — 专门为序列到序列不对齐问题设计的损失函数
    criterion = nn.CTCLoss(
        blank=0,           # blank token 的索引 = 0
        zero_infinity=True,  # 防止无限 loss
        reduction='mean'    # 对 batch 取平均
    )
    print(f"  损失函数: CTCLoss (blank_idx=0)")

    # ---- 第4步: 训练循环 ----
    print(f"\n[4/5] 开始训练...")
    print(f"  {'='*50}")

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'epochs': []}

    for epoch in range(TRAIN_CONFIG.NUM_EPOCHS):
        # 训练一个 epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch)

        # 验证
        val_loss = validate(model, val_loader, criterion)

        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)

        # ---- 保存最好的模型 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            save_path = os.path.join(SAVED_MODELS_DIR, 'exocet_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'charset': list(IDX2CHAR.values()),
            }, save_path)
            print(f"    💾 保存最佳模型: {save_path}")
        else:
            patience_counter += 1

        # ---- 检查点保存 ----
        if (epoch + 1) % TRAIN_CONFIG.SAVE_EVERY == 0:
            ckpt_path = os.path.join(SAVED_MODELS_DIR,
                                     f'exocet_checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, ckpt_path)

        # ---- 早停 (Early Stopping) ----
        # 如果连续 N 个 epoch 验证 loss 没降，就停止训练
        # 防止过拟合 + 节省时间
        if TRAIN_CONFIG.EARLY_STOP_ENABLED and patience_counter >= TRAIN_CONFIG.EARLY_STOP_PATIENCE:
            print(f"\n  ⏹  早停! 连续 {TRAIN_CONFIG.EARLY_STOP_PATIENCE} 轮没有改善")
            break

    # ---- 第5步: 保存结果 ----
    print(f"\n[5/5] 保存训练记录...")

    # 训练日志
    log_path = os.path.join(OUTPUTS_DIR, 'training_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"  训练日志: {log_path}")

    # 最终模型
    final_path = os.path.join(SAVED_MODELS_DIR, 'exocet_final.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'charset': list(IDX2CHAR.values()),
    }, final_path)
    print(f"  最终模型: {final_path}")

    # ---- 训练总结 ----
    print(f"\n{'='*60}")
    print(f"  训练完成!")
    print(f"  最佳验证 Loss: {best_val_loss:.4f}")
    print(f"  初始验证 Loss: {history['val_loss'][0]:.4f}")
    improvement = (history['val_loss'][0] - best_val_loss) / history['val_loss'][0] * 100
    print(f"  改善:          {improvement:.1f}%")
    print(f"  模型保存在:     {SAVED_MODELS_DIR}/")
    print(f"{'='*60}")

    return model


if __name__ == "__main__":
    main()
