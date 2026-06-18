#!/bin/bash
# ============================================================================
# Tesseract_OCR + EasyOCR 微调环境安装脚本
# ============================================================================
# 用法:
#   bash setup.sh
#
# 说明:
#   1. 使用清华镜像源加速下载 (已写入 ~/.config/pip/pip.conf)
#   2. 设置 TMPDIR 到项目目录，避免 /tmp (tmpfs/RAM) 空间不足
#   3. GTX 1660 用 CUDA 版 PyTorch (cu121)
#
# 前置: 你的 C 盘只剩 2G，但 WSL 虚拟盘有 900G 空闲，不影响 pip 安装
# ============================================================================

echo "========================================"
echo "  EasyOCR 微调环境安装"
echo "========================================"

# ---- Step 0: 确保 pip 镜像源配置好了 ----
echo ""
echo "[Step 0] 检查 pip 镜像源..."
PIP_CONF="$HOME/.config/pip/pip.conf"
if grep -q "tsinghua" "$PIP_CONF" 2>/dev/null; then
    echo "  ✅ 清华大学镜像已配置"
else
    echo "  ⚠️  未检测到镜像配置，正在写入..."
    mkdir -p "$(dirname "$PIP_CONF")"
    cat > "$PIP_CONF" << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
    echo "  ✅ 镜像配置完成"
fi

# ---- Step 1: 创建项目内临时目录，避开 /tmp 空间限制 ----
echo ""
echo "[Step 1] 设置 TMPDIR..."
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_TMP="$PROJECT_DIR/.tmp"
mkdir -p "$LOCAL_TMP"
echo "  ✅ TMPDIR=$LOCAL_TMP"

# ---- Step 2: 升级 pip (可选但推荐) ----
echo ""
echo "[Step 2] 升级 pip..."
TMPDIR="$LOCAL_TMP" python3 -m pip install --no-cache-dir --upgrade pip -q 2>&1 | tail -1

# ---- Step 3: 安装 PyTorch (CUDA 版，最大包 ~800MB) ----
echo ""
echo "[Step 3] 安装 PyTorch (CUDA) 这是最大的包，请耐心等待..."
TMPDIR="$LOCAL_TMP" python3 -m pip install --no-cache-dir \
    torch torchvision 2>&1 | tail -5
echo "  ✅ PyTorch 安装完成"

# ---- Step 4: 安装 EasyOCR 和其他依赖 ----
echo ""
echo "[Step 4] 安装 EasyOCR + OpenCV..."
TMPDIR="$LOCAL_TMP" python3 -m pip install --no-cache-dir \
    easyocr opencv-python-headless 2>&1 | tail -5
echo "  ✅ EasyOCR 安装完成"

# ---- Step 5: 验证安装 ----
echo ""
echo "========================================"
echo "  验证安装"
echo "========================================"

python3 -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

python3 -c "
import easyocr
print(f'  EasyOCR:  {easyocr.__version__}')
"

python3 -c "
from PIL import Image
import numpy as np
print(f'  Pillow:   已安装')
print(f'  NumPy:    {np.__version__}')
"

echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "下一步: 运行各个阶段的脚本"
echo "  python3 stage1_baseline.py      ← 用预训练模型跑基线"
echo "  python3 stage2_generate_data.py  ← 用字体生成训练数据"
echo "  python3 stage3_model_anatomy.py  ← 解剖模型结构"
echo "  python3 stage4_finetune.py       ← 微调训练"
echo ""
