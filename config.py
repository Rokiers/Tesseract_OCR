"""
============================================================================
config.py —— 项目全局配置
============================================================================

所有阶段的脚本都从这里读取配置，修改一处即可全局生效。

修改指南:
  1. CHARSET: 从你的图片中收集所有出现过的字符，填到这里
  2. DATA_CONFIG: 控制生成多少张训练图
  3. TRAIN_CONFIG: 控制微调训练参数

============================================================================
"""

import os
import torch

# ============================================================
# 项目路径
# ============================================================

# 项目根目录 (config.py 所在目录)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 字体文件
FONT_PATH = os.path.join(PROJECT_ROOT, "exocet-blizzard-medium.ttf")

# 原始图片目录 (16张游戏截图)
REAL_IMAGES_DIR = os.path.join(PROJECT_ROOT, "physical_object")

# 生成的数据集目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# 模型保存目录
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

# 输出目录 (基线结果、评估报告)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# 预处理缓存目录
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "preprocessed")

# 确保所有目录存在
for _dir in [DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
             SAVED_MODELS_DIR, OUTPUTS_DIR, PREPROCESSED_DIR]:
    os.makedirs(_dir, exist_ok=True)


ALLOWED_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ :+()-%'."
# ============================================================
# 字符集 (字符 → 序号 映射)
# ============================================================
# 重要: 从这个列表来确定模型识别的字符范围
#       你在图片中看到哪些字符，就填哪些
#       CTC blank token 会自动添加在索引 0 位置
#
# 你需要在运行 stage2 之前修改这个列表!
# 从你的 16 张图片中收集所有出现的不同字符，填到下面:

CHARSET = [
    # --- 大写字母 ---
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',

    # --- 小写字母 ---
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z',

    # --- 数字 ---
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',

    # --- 标点和特殊符号 ---
    ' ', '+', '-', ':', '!', '.', ',', '/', '%', '"',
    "'", '(', ')', '[', ']',
]

# 构建映射表
CTC_BLANK = '[blank]'

# char → index (跳过 blank，blank 固定占位置 0)
CHAR2IDX = {ch: i + 1 for i, ch in enumerate(CHARSET)}  # 1-based
CHAR2IDX[CTC_BLANK] = 0  # blank 始终是索引 0

# index → char
IDX2CHAR = {i: ch for ch, i in CHAR2IDX.items()}

# 有效字符数 (不含 blank)
NUM_CHARS = len(CHARSET)

# 字符集字符串 (不带 blank，用于 EasyOCR 等库)
CHARSET_STR = ''.join(CHARSET)

print(f"[config] 字符集大小: {NUM_CHARS} 个字符 (不含 blank)")
print(f"[config] 字符集: {CHARSET_STR[:40]}...")


# ============================================================
# 数据生成配置
# ============================================================

class DataConfig:
    # 生成图片的宽度范围 (像素) —— 模拟不同长度的单词
    IMG_WIDTH_MIN = 64
    IMG_WIDTH_MAX = 320

    # 生成图片的高度 (像素) —— 固定，因为 CRNN 需要统一高度
    IMG_HEIGHT = 32

    # 字体大小 (点)
    FONT_SIZE_MIN = 20
    FONT_SIZE_MAX = 28

    # 训练集数量
    TRAIN_SAMPLES = 5000

    # 验证集数量
    VAL_SAMPLES = 1000

    # 测试集数量
    TEST_SAMPLES = 500

    # 语料来源: 从哪个词汇表取词
    # 可选: 'charset' (随机组合字符集的词) /
    #       'english' (英文常用词) /
    #       'custom' (自定义词表)
    CORPUS_MODE = 'charset'

    # 生成词汇的长度范围
    WORD_LEN_MIN = 3
    WORD_LEN_MAX = 15


DATA_CONFIG = DataConfig()


# ============================================================
# 预处理配置
# ============================================================

class PreprocessConfig:
    # 是否转为灰度图
    TO_GRAY = True

    # 二值化阈值 (Otsu 自动 = None, 手动 = 0~255)
    BINARY_THRESHOLD = None  # None = 使用 Otsu 自动阈值

    # 对比度拉伸 (CLAHE)
    CLAHE_ENABLED = True
    CLAHE_CLIP_LIMIT = 2.0

    # 缩放后目标高度
    TARGET_HEIGHT = 32


PREPROCESS_CONFIG = PreprocessConfig()


# ============================================================
# 模型训练配置
# ============================================================

class TrainConfig:
    # 设备: 'auto' / 'cuda' / 'cpu'
    DEVICE = 'auto'  # 自动检测 GPU

    # 输入图片高度 (必须和模型结构一致)
    IMG_HEIGHT = 32

    # 批次大小 (GTX 1660 6GB 可以用 64)
    BATCH_SIZE = 64

    # 训练轮数
    NUM_EPOCHS = 10

    # 学习率
    LEARNING_RATE = 0.001

    # 优化器: 'adam' / 'sgd'
    OPTIMIZER = 'adam'

    # 每 N 个 epoch 保存一次模型
    SAVE_EVERY = 2

    # 验证频率 (每 N step)
    VAL_EVERY = 200

    # 冻结 CNN 前几层
    # None = 全部训练
    # int   = 冻结前 N 个卷积层
    FREEZE_CNN_LAYERS = 4  # 冻结前 4 层卷积 (方案C)

    # 早停 (Early Stopping)
    EARLY_STOP_PATIENCE = 5  # N 轮验证损失不降就停止
    EARLY_STOP_ENABLED = True

    # 数据增强
    # 暂时不做，等需要时再开启
    AUGMENTATION_ENABLED = False

    # 梯度裁剪 (防止梯度爆炸)
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
