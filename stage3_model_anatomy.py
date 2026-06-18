"""
============================================================================
阶段3: 模型解剖 —— 逐层拆解 CRNN 识别器的内部结构
============================================================================

目的:
  在开始训练之前，先搞清楚这个「大脑」长什么样。
  我们会把 EasyOCR 的预训练识别器加载出来，逐层打印:
    - 每层叫什么名字？是什么类型的层？
    - 每层有多少参数？（可训练 vs 不可训练）
    - CNN / RNN / Head 各占多少参数？

CRNN 模型的三层结构回顾:
  输入图片 (B, 1, 32, W)
        │
        ▼
  ┌─────────────────────────────────┐
  │ CNN 卷积层 (FeatureExtraction)    │  ← 提取视觉特征
  │  Conv2d → BN → ReLU → Pool → ... │     "哪里有笔画？"
  │  输出: (B, 512, 1, W/4)          │
  └───────────────┬─────────────────┘
                  ▼
  ┌─────────────────────────────────┐
  │ RNN 循环层 (SequenceModeling)     │  ← 理解上下文
  │  双向GRU                         │     "这个特征序列像什么字？"
  │  输出: (B, W/4, 256)             │
  └───────────────┬─────────────────┘
                  ▼
  ┌─────────────────────────────────┐
  │ Head 分类头 (Prediction)          │  ← 做出最终决策
  │  Linear(256, 97)                 │     "这一帧 = 字符'a'"
  │  输出: (B, W/4, 97)              │
  └─────────────────────────────────┘

核心概念速查:
  · 卷积核 (Kernel):  小矩阵在图片上滑动，检测局部特征
    比如一个 3x3 卷积核可以检测「左上到右下的斜线」
  · 感受野 (Receptive Field):  一个神经元能看到多大局部的原始图片
    第1层: 3x3 像素
    第5层: 可能看到 60x60 像素
    第10层: 可能看到整张图
  · 池化 (Pooling):  缩小特征图，保留核心信息，减少计算量
  · 隐藏状态 (Hidden State):  RNN 的记忆，把过去看到的信息编码进一个向量
  · GRU 门控:  更新门(Z_t)决定记住多少新信息，重置门(R_t)决定忘掉多少旧信息
  · CTC Loss:  不需要像素-字符对齐，自动学习「哪些帧属于哪个字符」
    blank token: 一个特殊标记「ε」，表示「这里暂时没有字符」

输出:
  直接在终端打印模型结构 + 参数量分析
============================================================================
"""

import os
import sys

import torch
import torch.nn as nn
import easyocr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, DEVICE, NUM_CHARS


# ============================================================
# 第1步: 加载模型
# ============================================================

def load_model():
    """
    加载 EasyOCR 识别器模型。

    EasyOCR 的流水线:
      Reader → Detector + Recognizer
                       ↑
                 我们只用这个来解剖

    识别器内部:
      model.FeatureExtraction  ← CNN
      model.SequenceModeling   ← RNN
      model.Prediction         ← Head
    """
    print("=" * 60)
    print("  阶段3: CRNN 模型解剖")
    print("=" * 60)

    print(f"\n[1/4] 加载 EasyOCR 识别器...")
    reader = easyocr.Reader(
        ['en'],
        gpu=(DEVICE.type == 'cuda'),
        model_storage_directory=os.path.join(PROJECT_ROOT, 'easyocr_models'),
        verbose=False
    )

    # 拿到识别器模型
    model = reader.recognizer.model

    # 移到对应设备
    model = model.to(DEVICE)
    model.eval()  # 评估模式（不计算梯度，节省内存）

    print(f"  模型类型: {type(model).__name__}")
    print(f"  设备:     {DEVICE}")
    return model


# ============================================================
# 第2步: 逐层探索 CNN 结构
# ============================================================

def explore_cnn(model):
    """
    探索 CNN 部分的结构。

    CNN 由多个卷积层堆叠而成，每层学到的特征越来越抽象:

    第 1~3 层: 低级特征
      - 边缘检测 (edge detection)
      - 颜色/亮度梯度
      - 笔画方向

    第 4~7 层: 中级特征
      - 拐角、圆角
      - 笔画的交叉点
      - 文字的局部形状

    第 8~12 层: 高级特征
      - 完整的笔画组合
      - 特定字符的视觉模式
      - "这一列像素像不像字母 'a' 的一部分？"
    """
    print(f"\n{'='*60}")
    print(f"  【CNN 卷积层 —— 视觉特征提取】")
    print(f"{'='*60}")
    print(f"""
  CNN (卷积神经网络) 的核心思想:

  卷积核 (Kernel / Filter):
  ┌───┬───┬───┐
  │-1 │ 0 │ 1 │  ← 一个 3x3 的 Sobel 边缘检测核
  ├───┼───┼───┤     在图片上水平滑动，每停一次就做一次
  │-2 │ 0 │ 2 │     矩阵乘法: kernel * 图片局部 = 一个数字
  ├───┼───┼───┤     所有位置的结果拼成一张「特征图」
  │-1 │ 0 │ 1 │
  └───┴───┴───┘

  一个卷积层的参数个数:
    Conv2d(in_channels=64, out_channels=128, kernel_size=3)
    = 64 × 128 × 3 × 3 + 128(bias)
    ≈ 73,856 个参数

  池化 (Pooling) — 降采样:
    把 2x2 的区域压缩成 1 个值（取最大 = MaxPool，取平均 = AvgPool）
    效果: 特征图尺寸减半，但核心特征保留
          感受野扩大（更深层的神经元能「看到」更大的原始图片区域）
    """)

    total_params = 0

    # 逐层打印 CNN
    cnn = model.FeatureExtraction

    print(f"\n  CNN 总结构: {cnn}")
    print(f"\n  {'层名':<40s} {'类型':<25s} {'输出尺寸':<25s} {'参数量':<12s}")
    print(f"  {'-'*40} {'-'*25} {'-'*25} {'-'*12}")

    # 模拟一个输入，追踪每一层的输出形状变化
    dummy_input = torch.randn(1, 1, 32, 100).to(DEVICE)  # (B, C, H, W)
    x = dummy_input

    for name, module in cnn.named_children():
        with torch.no_grad():
            x_prev = x
            x = module(x)

        # 计算参数量
        params = sum(p.numel() for p in module.parameters())
        total_params += params

        # 格式打印
        layer_type = module.__class__.__name__
        output_shape = str(tuple(x.shape))

        print(f"  {name:<40s} {layer_type:<25s} {output_shape:<25s} {params:>8,d}")

    print(f"  {'-'*40} {'-'*25} {'-'*25} {'-'*12}")
    print(f"  CNN 总参数: {total_params:,d}")
    print(f"\n  输入:  {tuple(dummy_input.shape)}  (批次, 通道, 高度, 宽度)")
    print(f"  输出:  {tuple(x.shape)}  (已压平成时间序列)")

    print(f"""
  ⭐ 关键观察:
  - 高度从 32 逐步压缩到 1 (通过池化层)
    → 每列像素被压缩成一个特征向量
  - 通道数从 1 → 64 → 128 → 256 → 512
    → 每层提取越来越丰富的特征
  - 输出形状 (B, 512, 1, W) → 可以 reshape 为 (B, W, 512)
    → 变成时序序列，送入 RNN
    """)

    return total_params


# ============================================================
# 第3步: 探索 RNN 结构
# ============================================================

def explore_rnn(model):
    """
    探索 RNN 部分的结构。

    RNN 做什么?
      把 CNN 输出的特征序列，转化为「每个位置上最可能是哪个字符」的
      高层特征。它理解字符之间的上下文关系。

    为什么需要 RNN？
      假设 CNN 看到 "i" 和 "l" 的竖线区域:
        CNN:  "这里有一根竖线" (无法判断是 i 还是 l)
        RNN: "竖线 + 上面有个点 + 很窄 + 后面是字母 e"
              → 综合上下文判断: 很可能是 'i'
              → 因为 "ile" 出现概率远高于 "lle"

    GRU (Gated Recurrent Unit) — 一种简化的 LSTM:

    每一步的计算:
      R_t = σ(W_r · [H_{t-1}, X_t])     ← 重置门: 决定忘掉多少过去
      Z_t = σ(W_z · [H_{t-1}, X_t])     ← 更新门: 决定记住多少新信息
      H̃_t  = tanh(W · [R_t ⊙ H_{t-1}, X_t])  ← 候选隐藏状态
      H_t  = (1 - Z_t) ⊙ H_{t-1} + Z_t ⊙ H̃_t  ← 最终隐藏状态

    比喻:
      - R_t (重置门): 「咦，新主题开始了？」 → 忘掉旧话题
      - Z_t (更新门): 「这个新信息很重要！」 → 更新记忆
      - H_t (隐藏状态): 当前时刻的「理解」
    """
    print(f"\n{'='*60}")
    print(f"  【RNN 循环层 —— 上下文理解】")
    print(f"{'='*60}")

    rnn = model.SequenceModeling

    print(f"\n  RNN 总结构: {rnn}")
    print(f"\n  RNN 的核心概念:")

    print(f"""
  ┌─────────────────────────────────────┐
  │ 双向 GRU (Bidirectional GRU)         │
  │                                     │
  │  前向: 从左往右读                    │
  │  H₁→ H₂→ H₃→ ... → Hₙ              │
  │      每个 H 包含「当前位置+左边所有」  │
  │                                     │
  │  反向: 从右往左读                    │
  │  H₁← H₂← H₃← ... ← Hₙ              │
  │      每个 H 包含「当前位置+右边所有」  │
  │                                     │
  │  合并: [前向, 反向] 拼在一起          │
  │  → 每个位置都能看到完整的上下文!       │
  └─────────────────────────────────────┘

  为什么双向很重要？
    "I like apple"  →  单向前向看到 "I like" 时猜不到后面是 apple
                      双向能同时看到前后文，判断更准确
    """)

    total_params = 0

    print(f"  {'层名':<40s} {'类型':<25s} {'参数量':<12s}")
    print(f"  {'-'*40} {'-'*25} {'-'*12}")

    for name, module in rnn.named_children():
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        layer_type = module.__class__.__name__

        # 如果是 GRU，多打印一些细节
        extra = ""
        if isinstance(module, nn.GRU) or 'GRU' in str(type(module)):
            extra = f"(hidden_size={module.hidden_size}, bidirectional=True)"
        elif isinstance(module, nn.Linear):
            extra = f"(in={module.in_features}, out={module.out_features})"

        print(f"  {name:<40s} {layer_type:<25s} {params:>8,d}  {extra}")

    print(f"  {'-'*40} {'-'*25} {'-'*12}")
    print(f"  RNN 总参数: {total_params:,d}")

    return total_params


# ============================================================
# 第4步: 探索 Head 结构
# ============================================================

def explore_head(model):
    """
    探索 Head (分类头) 的结构。

    Head = 一个矩阵乘法层:
      输入: (B, seq_len, 256)  ← RNN 输出的特征
      输出: (B, seq_len, 97)   ← 97 个字符的得分

    本质就是:
      output = input @ W.T + b
      # W: (97, 256)  权重矩阵
      # b: (97,)       偏置向量

    「换 Head」就是换一个不同大小的 W 矩阵:
      - 97 类 → W: (97, 256)
      - 42 类 → W: (42, 256)
      参数更少，收敛更快

    CTC (Connectionist Temporal Classification) 解码:
      模型输出的不是「确定的一个字符」，而是
      「每个位置上，97 个字符各自的概率」

      然后 CTC 解码找出最可能的字符序列:
        概率最高的一条路径 → 合并连续相同字符 → 去掉 blank → 最终文字

      例如:
        帧1:  'H':0.9, 'e':0.05, 'l':0.02, ...
        帧2:  'ε':0.8, 'H':0.1,  ...  (ε = blank, 表示「这帧不是我」)
        帧3:  'e':0.95, ...
        帧4:  'ε':0.7, ...
        帧5:  'l':0.85, ...  (连续两个 l, 实际输出 "l")
        帧6:  'l':0.78, ...  (这个 l 和上一个合并)
        帧7:  'ε':0.6, ...
        帧8:  'o':0.92, ...

        → 解码: H-ε-e-ε-l-ε-o → "Hello"
    """
    print(f"\n{'='*60}")
    print(f"  【Head 分类头 —— 做出最终决策】")
    print(f"{'='*60}")

    head = model.Prediction

    print(f"\n  Head 结构: {head}")
    print(f"""
  Head 的本质:
    一个 Linear 层 = 矩阵乘法
    ┌──────────┐      ┌────┐     ┌──────────┐
    │ RNN特征   │  ×   │ Wᵀ │  +  │ 偏置 b   │  =  97个字符的得分
    │ (1, 256)  │      │97×256│    │  (97,)   │
    └──────────┘      └────┘     └──────────┘
    """)

    total_params = 0
    print(f"  {'层名':<40s} {'类型':<25s} {'参数量':<12s}")
    print(f"  {'-'*40} {'-'*25} {'-'*12}")

    for name, module in head.named_children():
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        layer_type = module.__class__.__name__

        extra = ""
        if isinstance(module, nn.Linear):
            extra = f"(in={module.in_features}, out={module.out_features})"

        print(f"  {name:<40s} {layer_type:<25s} {params:>8,d}  {extra}")

    print(f"  {'-'*40} {'-'*25} {'-'*12}")
    print(f"  Head 总参数: {total_params:,d}")

    print(f"""
  ⭐ 关键发现:
    97 个字符中，Head 负责把 256 维特征向量「翻译」成具体字符
    如果缩减字符集到 42 个:
      参数量从 {total_params:,d} → {head[0].in_features * 43 + 43 if hasattr(head, '__iter__') else '约一半'}
      CNN 和 RNN 不变，只换 Head
    """)

    if hasattr(head, '__iter__'):
        for m in head:
            if isinstance(m, nn.Linear):
                print(f"""
  你的字符集如果有 {NUM_CHARS} 个字符，Head 就是:
    nn.Linear({m.in_features}, {NUM_CHARS + 1})  ← +1 是 CTC blank
    = {m.in_features} × {NUM_CHARS + 1} + {NUM_CHARS + 1}
    = {m.in_features * (NUM_CHARS + 1) + (NUM_CHARS + 1):,d} 个参数
    """)
                break

    return total_params


# ============================================================
# 第5步: 总体参数分析
# ============================================================

def parameter_breakdown(model):
    """
    把所有层的参数汇总，画出饼图（文字版）。

    训练时我们冻结 CNN → 只有 RNN+Head 的参数会更新
    → 实际训练的参数远少于总参数
    """
    print(f"\n{'='*60}")
    print(f"  【总参数分析】")
    print(f"{'='*60}")

    # 统计各部分的参数
    cnn_params = sum(p.numel() for p in model.FeatureExtraction.parameters())
    rnn_params = sum(p.numel() for p in model.SequenceModeling.parameters())
    head_params = sum(p.numel() for p in model.Prediction.parameters())
    total = cnn_params + rnn_params + head_params

    print(f"""
  ┌─────────────────────────────────────────┐
  │ 总参数量分布                              │
  │                                         │
  │  CNN:  {cnn_params:>10,d}  ({cnn_params/total:>5.1%})  ─── 提取特征  │
  │  RNN:  {rnn_params:>10,d}  ({rnn_params/total:>5.1%})  ─── 上下文理解│
  │  Head: {head_params:>10,d}  ({head_params/total:>5.1%})  ─── 字符决策  │
  │  ─────────────────────────────          │
  │  总计: {total:>10,d}                     │
  └─────────────────────────────────────────┘
    """)

    # 训练时哪些参数会更新？
    print(f"  微调策略 (方案C — 冻结 CNN 前几层):")
    print(f"    冻结层:  CNN 前几层的卷积层")
    print(f"    训练层:  CNN 后几层 + RNN + Head")
    print(f"    ≈ {rnn_params + head_params + cnn_params // 2:,d} 个可训练参数")
    print(f"    ≈ {total - (rnn_params + head_params + cnn_params // 2):,d} 个冻结参数")

    # 和 LLM 对比，感受规模差异
    print(f"""
  📏 规模对比（直观感受）:
    CRNN (我们的):   ~{total/1e6:.1f}M 参数      (小模型，CPU 也能训)
    BERT:            ~110M 参数                  (中等模型)
    GPT-4:           ~1.76T 参数                 (巨型模型)
    DeepSeek-V3:     ~671B 参数                  (巨型模型)

    我们的模型只有 {total/1e6:.1f}M 参数，6GB 显存绰绰有余！
    """)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    # 加载模型
    model = load_model()

    # 解剖三层结构
    cnn_p = explore_cnn(model)
    rnn_p = explore_rnn(model)
    head_p = explore_head(model)

    # 总体分析
    parameter_breakdown(model)

    print(f"\n{'='*60}")
    print("  阶段3 完成！")
    print(f"  → 你现在理解了 CRNN 的三层结构")
    print(f"  → 下一步: python3 stage4_finetune.py")
    print(f"{'='*60}")
