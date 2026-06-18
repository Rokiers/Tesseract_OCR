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
    PROJECT_ROOT, FONT_PATH, CHARSET, CHARSET_STR,
    DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    DATA_CONFIG
)


# ============================================================
# 第1步: 构建语料库（哪些词用来渲染）
# ============================================================

def build_corpus() -> list:
    """
    构建渲染语料库。

    语料来源:
      1. 随机字符组合（默认）: 从你的字符集中随机组合出「伪单词」
         比如 "ARMO", "SWRD", "+12 DEF" 这种暗黑风格词
      2. 英文常用词: 用 NLTK 词表 / 维基词频
      3. 自定义词表: 你手动指定（游戏装备词条等）

    为什么用字符集随机组合？
      因为这是最小可行方式。你还没确认图片里具体有哪些词，
      用字符集随机组合能确保覆盖所有字符。
      后续你可以替换成真实的游戏词表。
    """
    words = set()

    # 方案1: 从字符集随机生成伪单词
    # 对于每个单词长度 (3~15)，生成一些随机组合
    for length in range(DATA_CONFIG.WORD_LEN_MIN, DATA_CONFIG.WORD_LEN_MAX + 1):
        # 每种长度生成一定数量的词
        num_per_length = DATA_CONFIG.TRAIN_SAMPLES // (DATA_CONFIG.WORD_LEN_MAX - DATA_CONFIG.WORD_LEN_MIN + 1)
        for _ in range(num_per_length):
            # 从字符集中随机选 length 个字符组成单词
            word = ''.join(random.choices(CHARSET, k=length)).strip()
            if word:  # 排除空白词
                words.add(word)

    # 方案2: 添加一些英文常用词（如果装了 NLTK）
    # 这样模型能学到常见英文单词的上下文
    try:
        # 尝试加载英文词库（这是 Python 标准库自带！）
        # /usr/share/dict/words 在大多数 Linux 系统都存在
        if os.path.exists('/usr/share/dict/words'):
            with open('/usr/share/dict/words', 'r') as f:
                dict_words = [
                    w.strip() for w in f.readlines()
                    if w.strip().isalpha()
                    and len(w.strip()) >= DATA_CONFIG.WORD_LEN_MIN
                    and len(w.strip()) <= DATA_CONFIG.WORD_LEN_MAX
                ]
                # 只保留由我们字符集中字符组成的词
                allowed_chars = set(CHARSET_STR)
                filtered = [w for w in dict_words if all(c in allowed_chars for c in w)]
                words.update(random.sample(filtered, min(2000, len(filtered))))
                print(f"  ✅ 加载了 {len(filtered)} 个匹配字符集的英文词")
    except Exception:
        pass  # 没有 dict 文件也不影响，随机组合足够了

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
    # 创建空白图片（白色背景 = 255）
    img = Image.new('L', (width, height), color=255)  # 'L' = 灰度模式

    # 获取绘图上下文
    draw = ImageDraw.Draw(img)

    # ----- 计算文字在图片中的位置 -----
    # getbbox() 返回 (x0, y0, x1, y1) = 文字的外接矩形
    bbox = draw.textbbox((0, 0), word, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 如果文字太宽，调整图片宽度
    if text_width > width - 4:  # 留 4px 边距
        width = text_width + 4

    # 居中放置文字
    x = (width - text_width) // 2
    # 垂直居中: 考虑基线上方的 ascent
    y = (height - text_height) // 2 - bbox[1]  # bbox[1] 可能是负的（above baseline）

    # 如果居中后溢出，重新创建图片
    if x < 0 or y < 0:
        img = Image.new('L', (max(width, text_width + 4), height), color=255)
        draw = ImageDraw.Draw(img)
        x = max(2, x)
        y = max(2, y)

    # ----- 绘制文字 -----
    # 黑色文字 (color=0)，白色背景 (color=255)
    draw.text((x, y), word, font=font, fill=0)

    return np.array(img), word


# ============================================================
# 第3步: 生成数据集
# ============================================================

def generate_dataset(output_dir: str, words: list, num_samples: int, split_name: str):
    """
    生成一份数据集（训练/验证/测试）。

    流程:
      for i in range(num_samples):
        1. 随机选一个词
        2. 随机选字体大小（模拟不同距离/分辨率的文字）
        3. 随机选图片宽度（模拟不同长度的单词）
        4. 渲染 → 保存为 PNG
        5. 记录 label → labels.txt

    labels.txt 格式（每行一个样本）:
      train/00001.png ARMOR
      train/00002.png SWORD
      ...
    """
    os.makedirs(output_dir, exist_ok=True)

    labels = []  # [(图片名, 文字), ...]

    print(f"\n  生成 {split_name} 集 ({num_samples} 张)...")

    for i in range(num_samples):
        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{num_samples}]")

        # ----- 随机选一个词 -----
        word = random.choice(words)

        # ----- 随机字体大小 -----
        # 不同大小模拟不同字号 —— 就像游戏里不同位置的文字大小可能不同
        font_size = random.randint(DATA_CONFIG.FONT_SIZE_MIN,
                                    DATA_CONFIG.FONT_SIZE_MAX)
        font = ImageFont.truetype(FONT_PATH, size=font_size)

        # ----- 随机图片宽度 -----
        # 不同宽度模拟不同长度的单词 —— 模型需要学会处理各种宽度
        width = random.randint(DATA_CONFIG.IMG_WIDTH_MIN,
                               DATA_CONFIG.IMG_WIDTH_MAX)

        # ----- 渲染 -----
        img, label = render_word(word, font, width, DATA_CONFIG.IMG_HEIGHT)

        # ----- 保存 -----
        filename = f"{i:05d}.png"
        filepath = os.path.join(output_dir, filename)
        Image.fromarray(img).save(filepath)
        labels.append(f"{filename}\t{label}")

    # ----- 写 labels 文件 -----
    labels_path = os.path.join(output_dir, 'labels.txt')
    with open(labels_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))

    # ----- 写配置信息 -----
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

    # 验证集 — 用不同的词避免「作弊」
    val_words = [w for w in words if w not in set(
        open(os.path.join(TRAIN_DIR, 'labels.txt')).read().split('\t')[1]
        for _ in range(DATA_CONFIG.TRAIN_SAMPLES)
    )]
    # 简化: 用不同的随机种子选词
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
