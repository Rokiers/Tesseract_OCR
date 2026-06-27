"""
============================================================================
阶段2: 训练数据生成 —— 用 Exocet 字体渲染标注数据
============================================================================

目的:
  机器学习 = 数据 + 模型 + 训练。没有标注数据就没法学。
  我们有 .ttf 字体文件 → 可以程序化渲染「无限量」标注数据。

原理: 为什么合成数据有效？

  1. OCR 的难点在于「字体/字形变化」，不是「图片背景」
  2. 用你的字体渲染文字 → 模型能学到 Exocet 字体的笔画特征
  3. 每个生成图片的 label 是精确已知的（我们渲染的文字本身）
  4. 这就是 EasyOCR 预训练模型当初的训练方式 — 只是他们用了 900 万张图

数据集划分: 为什么是三份？
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ 训练集     │  │ 验证集     │  │ 测试集      │
  │ (train)   │  │ (val)     │  │ (test)     │
  │ 80%       │  │ 10%       │  │ 10%        │
  │ 用来更新参数│  │ 调整超参   │  │ 「期末考试」 │
  │ 「做作业」  │  │ 「月考」   │  │ 最终评估    │
  └──────────┘  └──────────┘  └──────────┘
  训练集≠测试集很重要！如果混在一起，模型只是「背答案」而非「理解」。
  这就是过拟合最直接的原因。

什么是过拟合？
  ┌─────────────────────────────────────────────────┐
  │  训练集准确率: 99%                                │
  │  测试集准确率: 45%  ← 这就是过拟合！              │
  │                                                 │
  │  模型背下了训练数据，但碰到没见过的图就傻眼了。      │
  │  就像你背了 10 道数学题答案，但考试换了个数字       │
  │  就不会做了。                                     │
  └─────────────────────────────────────────────────┘

输出:
  data/train/  ← 训练图片 + labels.txt
  data/val/    ← 验证图片 + labels.txt
  data/test/   ← 测试图片 + labels.txt
============================================================================
"""

import os
import sys
import random
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROJECT_ROOT, FONT_PATH, CHARSET_STR, NUM_CHARS,
    DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    DATA_CONFIG
)


# ============================================================
# 第1步: 构建语料库（哪些词用来渲染）
# ============================================================

def build_corpus() -> list:
    """
    构建渲染语料库。

    优先使用真实游戏词汇表 (vocabulary.txt)，包含:
      - 装备名 (1279个): TWO-HANDED SWORD, BUTCHER'S PUPIL...
      - 属性描述 (142个): FIRE RESIST, LIFE STOLEN PER HIT...
      - UI文本 (39个): ONE-HAND DAMAGE, SOCKETED, CTRL...

    如果词表不存在，回退到随机字符组合。
    """
    words = set()

    # 优先加载真实词表
    vocab_path = os.path.join(PROJECT_ROOT, 'vocabulary.txt')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and DATA_CONFIG.WORD_LEN_MIN <= len(word) <= DATA_CONFIG.WORD_LEN_MAX:
                    words.add(word)
        print(f"  📚 游戏词表: {len(words)} 个词")

    print(f"  ✅ 语料库共 {len(words)} 个词")
    return list(words)


# ============================================================
# 第2步: 渲染一张图片
# ============================================================

def render_word(word: str, font: ImageFont.FreeTypeFont, width: int, height: int) -> np.ndarray:
    """
    用指定的字体把文字渲染成灰度图片。

    参数:
      word:   要渲染的文字
      font:   PIL 字体对象（你的 Exocet 字体）
      width:  图片宽度（像素）
      height: 图片高度（像素）

    返回:
      numpy array, shape=(height, width), dtype=uint8, 值域 0~255

    核心概念: 为什么需要统一高度？
      CRNN 模型要求所有输入图片高度一致（H=32），因为 CNN 的
      卷积核需要在相同尺寸的特征图上滑动。但宽度可以不固定，
      因为 RNN 能处理任意长度序列。
    """
    # 先测量文字尺寸，再创建合适宽度的图片
    temp_img = Image.new('L', (1, 1), color=255)
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), word, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 确保图片至少能容纳文字 + 边距
    width = max(width, text_width + 4)

    # 创建空白图片（白色背景 = 255）
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)

    # 居中放置
    x = (width - text_width) // 2
    y = (height - text_height) // 2 - bbox[1]

    # 防止溢出
    x = max(2, x)
    y = max(2, y)

    # 绘制黑色文字
    draw.text((x, y), word, font=font, fill=0)

    return np.array(img), word


# ============================================================
# 第3步: 生成数据集
# ============================================================

def generate_dataset(output_dir: str, words: list, num_samples: int, split_name: str):
    """
    生成一份数据集（训练/验证/测试）。
    """
    os.makedirs(output_dir, exist_ok=True)

    labels = []

    print(f"\n  生成 {split_name} 集 ({num_samples} 张)...")

    for i in range(num_samples):
        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{num_samples}]")

        word = random.choice(words)

        font_size = random.randint(DATA_CONFIG.FONT_SIZE_MIN,
                                    DATA_CONFIG.FONT_SIZE_MAX)
        font = ImageFont.truetype(FONT_PATH, size=font_size)

        width = random.randint(DATA_CONFIG.IMG_WIDTH_MIN,
                                DATA_CONFIG.IMG_WIDTH_MAX)

        img, label = render_word(word, font, width, DATA_CONFIG.IMG_HEIGHT)

        filename = f"{i:05d}.png"
        filepath = os.path.join(output_dir, filename)
        Image.fromarray(img).save(filepath)
        labels.append(f"{filename}\t{label}")

    labels_path = os.path.join(output_dir, 'labels.txt')
    with open(labels_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))

    meta_path = os.path.join(output_dir, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'num_samples': num_samples,
            'img_height': DATA_CONFIG.IMG_HEIGHT,
            'font': os.path.basename(FONT_PATH),
            'charset': CHARSET_STR,
            'font_size_range': [DATA_CONFIG.FONT_SIZE_MIN, DATA_CONFIG.FONT_SIZE_MAX],
            'width_range': [DATA_CONFIG.IMG_WIDTH_MIN, DATA_CONFIG.IMG_WIDTH_MAX],
        }, f, ensure_ascii=False, indent=2)

    print(f"  ✅ {split_name}: {num_samples} 张 → {output_dir}/")


# ============================================================
# 第4步: 预览几张样本（人工检查质量）
# ============================================================

def preview_samples():
    """生成几张预览图，让人工检查渲染质量"""
    preview_dir = os.path.join(DATA_DIR, 'preview')
    os.makedirs(preview_dir, exist_ok=True)

    print(f"\n[预览] 生成 10 张样本用于人工检查...")

    font = ImageFont.truetype(FONT_PATH, size=24)
    test_words = [
        "ARMOR", "SWORD+1", "DEF:100", "ICE",
        "VAMPIRE", "Fire Ball", "Gold:999",
        "Helm", "Damage+50", "!Magic!"
    ]

    for i, word in enumerate(test_words):
        img, _ = render_word(word, font, 200, DATA_CONFIG.IMG_HEIGHT)
        filepath = os.path.join(preview_dir, f"preview_{i:02d}_{word}.png")
        Image.fromarray(img).save(filepath)
        print(f"  ✅ {filepath}")

    print(f"  → 请检查 preview/ 目录下的图片，确认字体渲染正常")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  阶段2: 训练数据生成")
    print("=" * 60)

    # ---- 检查字体文件 ----
    if not os.path.exists(FONT_PATH):
        print(f"\n❌ 字体文件不存在: {FONT_PATH}")
        sys.exit(1)
    print(f"\n✅ 字体文件: {FONT_PATH}")

    # ---- 检查字符集 ----
    print(f"✅ 字符集: {NUM_CHARS} 个字符")
    print(f"   ({CHARSET_STR[:60]}...)")

    # ---- 构建语料库 ----
    print(f"\n[1/4] 构建语料库...")
    random.seed(42)  # 固定随机种子，保证可复现
    words = build_corpus()

    # ---- 预览样本 ----
    print(f"\n[2/4] 预览渲染效果...")
    preview_samples()

    # ---- 生成数据集 ----
    print(f"\n[3/4] 生成数据集...")
    print(f"  训练集: {DATA_CONFIG.TRAIN_SAMPLES} 张")
    print(f"  验证集: {DATA_CONFIG.VAL_SAMPLES} 张")
    print(f"  测试集: {DATA_CONFIG.TEST_SAMPLES} 张")

    # 训练集
    generate_dataset(TRAIN_DIR, words, DATA_CONFIG.TRAIN_SAMPLES, "训练集")

    # 验证集
    random.seed(123)
    generate_dataset(VAL_DIR, words, DATA_CONFIG.VAL_SAMPLES, "验证集")

    # 测试集
    random.seed(456)
    generate_dataset(TEST_DIR, words, DATA_CONFIG.TEST_SAMPLES, "测试集")

    # ---- 统计信息 ----
    print(f"\n[4/4] 数据集统计")
    total = DATA_CONFIG.TRAIN_SAMPLES + DATA_CONFIG.VAL_SAMPLES + DATA_CONFIG.TEST_SAMPLES
    print(f"  总计: {total} 张图片")
    print(f"  训练: {DATA_CONFIG.TRAIN_SAMPLES} 张 ({DATA_CONFIG.TRAIN_SAMPLES/total:.0%})")
    print(f"  验证: {DATA_CONFIG.VAL_SAMPLES} 张 ({DATA_CONFIG.VAL_SAMPLES/total:.0%})")
    print(f"  测试: {DATA_CONFIG.TEST_SAMPLES} 张 ({DATA_CONFIG.TEST_SAMPLES/total:.0%})")

    print(f"\n{'='*60}")
    print("  阶段2 完成！")
    print(f"  → 训练数据: {TRAIN_DIR}/")
    print(f"  → 验证数据: {VAL_DIR}/")
    print(f"  → 测试数据: {TEST_DIR}/")
    print(f"  → 预览样本: {DATA_DIR}/preview/")
    print(f"  → 下一步: python3 stage3_model_anatomy.py")
    print(f"{'='*60}")
