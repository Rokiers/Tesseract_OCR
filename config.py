"""
============================================================================
config.py —— 项目全局配置
============================================================================
"""

import os
import torch

# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(PROJECT_ROOT, "exocet-blizzard-medium.ttf")
REAL_IMAGES_DIR = os.path.join(PROJECT_ROOT, "physical_object")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "preprocessed")

for _dir in [DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
             SAVED_MODELS_DIR, OUTPUTS_DIR, PREPROCESSED_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ============================================================
# 字符集
# ============================================================

CHARSET = [
    # 数字
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    # 大写字母
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    # 符号 (全部来自游戏实际显示)
    ' ', '+', '(', ')', '-', '%', "'", '.', ':',
]

CTC_BLANK = '[blank]'

CHAR2IDX = {ch: i + 1 for i, ch in enumerate(CHARSET)}
CHAR2IDX[CTC_BLANK] = 0

IDX2CHAR = {i: ch for ch, i in CHAR2IDX.items()}

NUM_CHARS = len(CHARSET)
CHARSET_STR = ''.join(CHARSET)

print(f"[config] 字符集大小: {NUM_CHARS} 个字符 (不含 blank)")
print(f"[config] 字符集: {CHARSET_STR}")


# ============================================================
# 数据生成配置
# ============================================================

class DataConfig:
    IMG_WIDTH_MIN = 64
    IMG_WIDTH_MAX = 500
    IMG_HEIGHT = 32
    FONT_SIZE_MIN = 20
    FONT_SIZE_MAX = 28
    TRAIN_SAMPLES = 8000
    VAL_SAMPLES = 1500
    TEST_SAMPLES = 800
    WORD_LEN_MIN = 2
    WORD_LEN_MAX = 55


DATA_CONFIG = DataConfig()


# ============================================================
# 预处理配置
# ============================================================

class PreprocessConfig:
    TO_GRAY = True
    BINARY_THRESHOLD = None
    CLAHE_ENABLED = True
    CLAHE_CLIP_LIMIT = 2.0
    TARGET_HEIGHT = 32


PREPROCESS_CONFIG = PreprocessConfig()


class TrainConfig:
    DEVICE = 'auto'
    IMG_HEIGHT = 32
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    SAVE_EVERY = 2
    FREEZE_CNN_LAYERS = 4
    EARLY_STOP_PATIENCE = 5
    EARLY_STOP_ENABLED = True
    GRAD_CLIP = 5.0


TRAIN_CONFIG = TrainConfig()


# ============================================================
# 设备自动检测
# ============================================================

def get_device() -> torch.device:
    """自动检测并返回训练设备 (GPU > CPU)"""
    if TRAIN_CONFIG.DEVICE == 'cpu':
        return torch.device('cpu')
    if TRAIN_CONFIG.DEVICE == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("[config] 警告: 配置要求 CUDA 但不可用，回退到 CPU")
            return torch.device('cpu')
    # auto
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[config] 使用 GPU: {gpu_name} ({vram_gb:.1f} GB)")
        return torch.device('cuda')
    else:
        print("[config] 使用 CPU")
        return torch.device('cpu')


DEVICE = get_device()


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("配置检查完成")
    print(f"  项目路径:  {PROJECT_ROOT}")
    print(f"  字体文件:  {FONT_PATH}  {'✅' if os.path.exists(FONT_PATH) else '❌ 不存在!'}")
    print(f"  真实图片:  {REAL_IMAGES_DIR}")
    print(f"  数据目录:  {DATA_DIR}")
    print(f"  模型目录:  {SAVED_MODELS_DIR}")
    print(f"  训练设备:  {DEVICE}")
    print(f"  字符集:    {NUM_CHARS} 个字符")
    print(f"{'='*60}")
