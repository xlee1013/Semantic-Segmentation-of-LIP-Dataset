import torch
import numpy as np
import os
import os.path as osp
import settings


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


def ensure_dir(dir_path):
    if not osp.isdir(dir_path):
        os.makedirs(dir_path)


def write_train_log_txt_file(dic):
    with open("logdir/train.txt", "a") as f:
        info = "step: %d, lr: %.5f, loss: %.5f\n" % (dic['step'], dic['lr'], dic['loss'])
        f.write(info)
        f.close()


from PIL import Image
import matplotlib.pyplot as plt

PALETTE_20 = [0, 0, 0,  # 调色板，20类别
             128, 0, 0,
             0, 128, 0,
             128, 128, 0,
             0, 0, 128,
             128, 0, 128,
             0, 128, 128,
             128, 128, 128,
             64, 0, 0,
             192, 0, 0,
             64, 128, 0,
             192, 128, 0,
             64, 0, 128,
             192, 0, 128,
             64, 128, 128,
             192, 128, 128,
             0, 64, 0,
             128, 64, 0,
             0, 192, 0,
             128, 192, 0]
PALETTE_02 = [255, 255, 255,  # 调色板，2个类别，背景全白，前景全黑
             0, 0, 0]


def save_pred_pic(mask, mask_name, mode="eval", show=False):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    if settings.N_CLASSES == 20:
        new_mask.putpalette(PALETTE_20)
    if settings.N_CLASSES == 2:
        new_mask.putpalette(PALETTE_02)
    if show:
        plt.figure()
        plt.imshow(new_mask)
        plt.show()
    if mode == "eval":
        save_root = settings.SAVE_EVAL_PRED_PATH.format(settings.N_CLASSES, mask_name)
    else:
        save_root = settings.SAVE_TEST_PRED_PATH.format(settings.N_CLASSES, mask_name)
    new_mask.save(save_root)
