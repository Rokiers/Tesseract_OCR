"""
============================================================================
阶段1: 基线测试 —— 用 EasyOCR 预训练模型直跑 16 张真实图片
============================================================================

目的:
  1. 感受 OCR 的输入/输出：图片进去，文字出来，置信度多少
  2. 确认哪些字符被混淆（i/1/j、0/o 等）
  3. 建立一个可量化的 baseline（基线准确率），后续微调才有参照

OCR 的两个步骤（EasyOCR 自动处理，你只需要知道原理）:
  ┌──────────┐      ┌───────────┐
  │ 文本检测   │  →  │ 文本识别   │  →  输出文字
  │ (Detector)│      │(Recognizer)│
  └──────────┘      └───────────┘
  找到图中哪里有文字    把文字区域转成字符

输出:
  - outputs/baseline_results.txt   ← 每张图的识别结果
  - outputs/baseline_comparison/    ← 原图 + 标注框的对比图
============================================================================
"""

import os
import sys
import json
import time
from datetime import datetime

import cv2
import numpy as np
import easyocr
from pathlib import Path

# 导入项目配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROJECT_ROOT, REAL_IMAGES_DIR, OUTPUTS_DIR, DEVICE,
    CHARSET_STR, NUM_CHARS, PREPROCESS_CONFIG
)


# ============================================================
# 第1步: 图片预处理（让文字更清晰）
# ============================================================

def preprocess_image(image_path: str) -> np.ndarray:
    """
    对游戏截图做预处理，让文字区域更清晰。

    预处理流程:
      原图 → 灰度化 → CLAHE 对比度增强 → 输出

    CLAHE (Contrast Limited Adaptive Histogram Equalization):
      局部自适应直方图均衡化。和普通直方图均衡化的区别在于:
      - 普通: 全局调整，可能导致亮度区域过曝
      - CLAHE: 把图片分成小块，每块独立调整直方图，再拼回去
        这样暗处的文字也能被"提亮"，亮处的文字也不会过曝

    为什么不做二值化？
      二值化会把像素强分成 0/255，可能切断文字的细笔画。
      EasyOCR 的 CNN 更喜欢灰度图（保留了边缘过渡信息）。
    """
    # 用 OpenCV 读取（保留中文路径兼容性）
    img = cv2.imdecode(
        np.fromfile(image_path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        print(f"  ⚠️  无法读取: {image_path}")
        return None

    # 灰度化 — 把 RGB 三通道变成单通道，减少信息冗余
    # CNN 对颜色的依赖几乎为 0（文字识别主要靠形状，不是颜色）
    if PREPROCESS_CONFIG.TO_GRAY:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE 对比度增强
    if PREPROCESS_CONFIG.CLAHE_ENABLED:
        clahe = cv2.createCLAHE(
            clipLimit=PREPROCESS_CONFIG.CLAHE_CLIP_LIMIT,
            tileGridSize=(8, 8)  # 分成 8x8 的小块
        )
        img = clahe.apply(img)

    return img


# ============================================================
# 第2步: 用 EasyOCR 做识别
# ============================================================

def run_baseline():
    """
    主流程:
      1. 初始化 EasyOCR Reader（加载预训练模型）
      2. 遍历 16 张图，逐个识别
      3. 保存结果到文本文件
      4. 生成带标注框的对比图
    """

    print("=" * 60)
    print("  阶段1: EasyOCR 基线测试")
    print("=" * 60)

    # ---------- 2.1 初始化 Reader ----------

    print(f"\n[1/3] 加载 EasyOCR 预训练模型...")
    print(f"  设备: {DEVICE}")
    print(f"  语言: English (en)")
    print(f"  这是从 ~/.EasyOCR/model/ 下载的 .pth 权重文件")

    # Reader 会去下载预训练权重（约 180MB），如果已下载则跳过
    # gpu=True → 用 GPU 推理（检测器 + 识别器都在 GPU 上跑）
    reader = easyocr.Reader(
        ['en'],             # 语言: English
        gpu=(DEVICE.type == 'cuda'),
        model_storage_directory=os.path.join(PROJECT_ROOT, 'easyocr_models'),
        verbose=False
    )

    # ---------- 2.2 遍历图片 ----------

    print(f"\n[2/3] 开始识别 {REAL_IMAGES_DIR}/ 下的图片...\n")

    # 收集所有 PNG 图片
    image_files = sorted([
        f for f in os.listdir(REAL_IMAGES_DIR)
        if f.lower().endswith('.png')
    ])

    if not image_files:
        print("  ❌ 没找到 PNG 图片！")
        return

    # 所有结果存这里
    all_results = []
    comparison_dir = os.path.join(OUTPUTS_DIR, 'baseline_comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(REAL_IMAGES_DIR, filename)
        print(f"  [{idx:02d}/{len(image_files)}] {filename}")

        # 预处理
        img = preprocess_image(image_path)
        if img is None:
            continue

        # 计时
        t_start = time.time()

        # ----- 核心: EasyOCR 识别 -----
        # readtext() 返回: [(bbox, text, confidence), ...]
        #   bbox:       [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]  四个角点
        #   text:       "识别的文字"
        #   confidence: 0.0~1.0 置信度，越接近 1 越确定
        results = reader.readtext(
            img,
            low_text=0.05,
            detail=1,
            paragraph=False,
            contrast_ths=0.1,
            text_threshold=0.6,
            link_threshold=0.2
        )

        elapsed = time.time() - t_start

        # ----- 显示结果 -----
        detected_texts = []
        for item in results:
            if len(item) == 3:
                bbox, text, confidence = item
            elif len(item) == 2:
                bbox, text = item
                confidence = 1.0
            else:
                continue
            print(f"    [{confidence:.2%}] {text!r}")
            detected_texts.append({
                'text': text,
                'confidence': round(confidence, 4),
                'bbox': [[int(p[0]), int(p[1])] for p in bbox]
            })

        all_results.append({
            'filename': filename,
            'num_detections': len(detected_texts),
            'time_seconds': round(elapsed, 3),
            'detections': detected_texts
        })

        # ----- 生成标注图（画框 + 文字）-----
        draw_annotations(image_path, results, comparison_dir)

    # ---------- 2.3 保存结果 ----------

    print(f"\n[3/3] 保存结果...")

    # 文本报告
    report_path = os.path.join(OUTPUTS_DIR, 'baseline_results.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"EasyOCR 基线测试报告\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        for r in all_results:
            f.write(f"--- {r['filename']} ---\n")
            for det in r['detections']:
                f.write(f"  [{det['confidence']:.2%}] {det['text']!r}\n")
            f.write("\n")

    # JSON 详细结果
    json_path = os.path.join(OUTPUTS_DIR, 'baseline_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"  文本报告: {report_path}")
    print(f"  JSON 数据: {json_path}")

    return all_results


# ============================================================
# 辅助: 画标注框
# ============================================================

def draw_annotations(image_path: str, results: list, out_dir: str):
    """在原图上画检测框和识别文字，保存对比图"""
    img = cv2.imdecode(
        np.fromfile(image_path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    if img is None:
        return

    for item in results:
        if len(item) == 3:
            bbox, text, confidence = item
        elif len(item) == 2:
            bbox, text = item
            confidence = 1.0
        else:
            continue
        # 画四边形框
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        # 在框上方写字
        x, y = pts[0]
        label = f"{text} ({confidence:.0%})"
        cv2.putText(img, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{basename}_annotated.png")
    cv2.imencode('.png', img)[1].tofile(out_path)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print(f"\n配置信息:")
    print(f"  项目目录: {PROJECT_ROOT}")
    print(f"  图片目录: {REAL_IMAGES_DIR}")
    print(f"  字符集:   {NUM_CHARS} 个字符")
    print(f"  设备:     {DEVICE}")
    print()

    run_baseline()

    print(f"\n{'='*60}")
    print("  阶段1 完成！")
    print(f"  → 查看识别结果: outputs/baseline_results.txt")
    print(f"  → 查看标注图片: outputs/baseline_comparison/")
    print(f"  → 下一步: python3 stage2_generate_data.py")
    print(f"{'='*60}")
