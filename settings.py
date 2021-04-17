import logging
import numpy as np
import torch
from torch import Tensor


# 数据设置
DATA_ROOT = 'LIP_DATA/'
SCALES = (0.75, 1.0, 1.25, 1.5)  # 输入CNN之前对图像缩放
CROP_SIZE = 256  # 训练时，每次将图像裁剪为 256 * 256
IGNORE_LABEL = 255  # VOC和 COCO数据集中像素值 255.0时代表是轮廓，不需要分类

# 预训练模型存放路径
VGG16_MODEL_PATH = "models/pretrained/vgg16-397923af.pth"  # VGG16
RESNET50_MODEL_PATH = "models/pretrained/resnet50-19c8e357.pth"  # ResNet50
RESNET18_MODEL_PATH = "models/pretrained/resnet18-5c106cde.pth"  # ResNet18

# 模型设置
CPU_OR_GPU = "cuda"  # 可选 ["cuda", "cpu"]，建议"cuda"
DEVICE = torch.device(CPU_OR_GPU)
BACKBONE = "ResNet"  # 可选["VGGNet", "ResNet"], 在这里使用的是 ResNet18, 建议使用 "ResNet"

N_CLASSES = 2  # 类别数，包含背景类, 可选[2, 20], 2是人与背景二分类，20是人体解析与背景20分类
STRIDE = 8  # 可选 [8, 16, 32]，决定使用 8s / 16s / 32s 模型

# 训练设置
BATCH_SIZE = 8  # batch size
EPOCHS = 20  # epochs
ITER_MAX = (30462 // BATCH_SIZE) * EPOCHS  # 最大迭代次数
ITER_SAVE = 2000  # 每训练 ITER_SAVE 轮保存一次模型参数
ITER_SAVE_LATSET = 100  # 每训练 ITER_SAVE_LATSET 轮保存一次最新的模型参数

LR_DECAY = 10  # 学习率衰减率
LR = 0.009  # 学习率
LR_MOM = 0.9  # 优化器SGD的动量
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4

# 分割图像保存
SAVE_EVAL_PRED_PATH = "results/classes_{}/eval/{}.png"  # 验证集保存路径
SAVE_TEST_PRED_PATH = "results/classes_{}/test/{}.png"  # 测试集保存路径

# 日志设置，不用管
LOG_DIR = './logdir' 
MODEL_DIR = './models/classes_{}/{}'.format(N_CLASSES, CPU_OR_GPU)

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
