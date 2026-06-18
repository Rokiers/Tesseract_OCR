"""
============================================================================
inference.py —— 最终推理脚本
============================================================================

用微调后的模型识别新图片。

用法:
  python3 inference.py physical_object/01.png           ← 单张图
  python3 inference.py physical_object/                  ← 整个目录
  python3 inference.py image.png --baseline              ← 对比微调前后
============================================================================
"""

import os
import sys
import argparse

import torch
import numpy as np
from PIL import Image
import easyocr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROJECT_ROOT, SAVED_MODELS_DIR, DEVICE,
    IDX2CHAR, CTC_BLANK, NUM_CHARS
)
from stage4_finetune import build_model


# ============================================================
# 加载微调后的模型
# ============================================================

def load_finetuned_model(model_path=None):
    """
    加载微调好的模型。

    如果没指定路径，自动找最新的:
      saved_models/exocet_best.pth > exocet_final.pth > checkpoint
    """
    if model_path is None:
        # 自动找最佳模型
        candidates = [
            os.path.join(SAVED_MODELS_DIR, 'exocet_best.pth'),
            os.path.join(SAVED_MODELS_DIR, 'exocet_final.pth'),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if model_path is None or not os.path.exists(model_path):
        print("❌ 找不到微调模型！请先运行 stage4_finetune.py")
        return None

    print(f"加载模型: {model_path}")

    # 构建模型
    model = build_model()

    # 加载权重
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(DEVICE)
    model.eval()

    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', '?'):.4f}")

    return model


# ============================================================
# CTC 解码
# ============================================================

def ctc_decode(indices):
    """合并重复 + 去 blank"""
    decoded = []
    prev = -1
    for idx in indices:
        if idx != prev and idx != 0:
            decoded.append(idx)
        prev = idx
    return decoded


# ============================================================
# 识别单张图
# ============================================================

def recognize_image(model, image_path, reader_baseline=None):
    """
    识别一张图片，返回文字。

    如果提供了 reader_baseline，还会对比微调前后的结果。
    """
    print(f"\n{'='*50}")
    print(f"图片: {os.path.basename(image_path)}")

    # ---- 预处理 ----
    from stage1_baseline import preprocess_image
    img = preprocess_image(image_path)
    if img is None:
        return

    # ---- 微调模型识别 ----
    # 将图片转为模型输入格式
    h, w = img.shape
    ratio = 32 / h
    new_w = max(int(w * ratio), 4)
    img_resized = np.array(Image.fromarray(img).resize(
        (new_w, 32), Image.BICUBIC), dtype=np.float32) / 255.0
    img_tensor = torch.FloatTensor(img_resized).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        # output shape: (B, T, num_classes)
        _, pred_indices = output.max(2)
        decoded = ctc_decode(pred_indices[0].cpu().numpy())

    # 索引 → 字符
    text = ''.join(IDX2CHAR.get(idx, '?') for idx in decoded)
    print(f"\n🔮 微调后: {text!r}")

    # ---- 基线模型对比（可选） ----
    if reader_baseline is not None:
        results = reader_baseline.readtext(img, detail=1, paragraph=False)
        if results:
            baseline_text = results[0][1]
            print(f"📊 基线:   {baseline_text!r}")
        else:
            print(f"📊 基线:   (无结果)")

    return text


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Exocet OCR 推理')
    parser.add_argument('input', help='图片路径或目录')
    parser.add_argument('--model', help='模型路径', default=None)
    parser.add_argument('--baseline', action='store_true',
                        help='同时对比预训练模型结果')
    args = parser.parse_args()

    # ---- 加载微调模型 ----
    model = load_finetuned_model(args.model)
    if model is None:
        return

    # ---- 加载基线模型（如果需要对比） ----
    reader = None
    if args.baseline:
        print("\n加载基线模型 (EasyOCR 预训练)...")
        reader = easyocr.Reader(
            ['en'],
            gpu=(DEVICE.type == 'cuda'),
            model_storage_directory=os.path.join(PROJECT_ROOT, 'easyocr_models'),
            verbose=False
        )

    # ---- 识别 ----
    input_path = args.input

    if os.path.isfile(input_path):
        # 单张图片
        recognize_image(model, input_path, reader)

    elif os.path.isdir(input_path):
        # 整个目录
        png_files = sorted([
            f for f in os.listdir(input_path) if f.lower().endswith('.png')
        ])
        if not png_files:
            print(f"目录 {input_path} 中没有 PNG 图片")
            return

        results = []
        for filename in png_files:
            img_path = os.path.join(input_path, filename)
            text = recognize_image(model, img_path, reader)
            results.append((filename, text))

        # 汇总
        print(f"\n{'='*50}")
        print(f"汇总 ({len(results)} 张图片):")
        for filename, text in results:
            print(f"  {filename:<20s} {text!r}")

    else:
        print(f"❌ 输入路径不存在: {input_path}")


if __name__ == "__main__":
    main()
