import os
import os.path as osp

from PIL import Image
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils import data

import settings


def fetch(image_path, label_path=None):
    """
    打开原图和标签，转换成 Tensor，并作图像标准化
    """
    with open(image_path, 'rb') as fp:
        image = Image.open(fp).convert('RGB')
    image = torch.FloatTensor(np.asarray(image)) / 255
    image_mean = torch.mean(image, dim=(0, 1), keepdim=True)
    image_std = torch.std(image, dim=(0, 1), keepdim=True)
    image = (image - image_mean) / image_std
    image = image.permute(2, 0, 1).unsqueeze(dim=0)

    if label_path is not None:
        with open(label_path, 'rb') as fp:
            label = Image.open(fp).convert('P')
        label = torch.FloatTensor(np.asarray(label))
        if settings.N_CLASSES == 20:
            pass
        else:
            label[label>0] = 1.0
            label[label==0] = 0.0
        label = label.unsqueeze(dim=0).unsqueeze(dim=1)
    else:
        label = None

    return image, label


def scale(image, label=None):
    """
    图像和标签同尺度随机缩放
    """
    ratio = np.random.choice(settings.SCALES)
    image = F.interpolate(image, scale_factor=ratio, mode='bilinear', 
                          align_corners=True)
    if label is not None:
        label = F.interpolate(label, scale_factor=ratio, mode='nearest')
    return image, label


def pad(image, label=None):
    """
    如果缩放完之后图像和标签的尺寸不足 256 * 256，则用 0 补足
    """
    h, w = image.size()[-2:] 
    crop_size = settings.CROP_SIZE
    pad_h = max(crop_size - h, 0)
    pad_w = max(crop_size - w, 0)
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0.)
        if label is not None:
            label = F.pad(label, (0, pad_w, 0, pad_h), mode='constant', 
                          value=settings.IGNORE_LABEL)
    return image, label


def pad_inf(image, label=None):
    h, w = image.size()[-2:] 
    stride = settings.STRIDE
    pad_h = (stride + 1 - h % stride) % stride
    pad_w = (stride + 1 - w % stride) % stride
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0.)
        if label is not None:
            label = F.pad(label, (0, pad_w, 0, pad_h), mode='constant', 
                          value=settings.IGNORE_LABEL)
    return image, label


def crop(image, label=None):
    """
    对图像和标签做随机裁剪
    """
    h, w = image.size()[-2:]
    crop_size = settings.CROP_SIZE
    s_h = np.random.randint(0, h - crop_size + 1)
    s_w = np.random.randint(0, w - crop_size + 1)
    e_h = s_h + crop_size
    e_w = s_w + crop_size
    image = image[:, :, s_h: e_h, s_w: e_w]
    label = label[:, :, s_h: e_h, s_w: e_w]
    return image, label


def flip(image, label=None):
    """
    对图像和标签做随机翻转
    """
    if np.random.rand() < 0.5:
        image = torch.flip(image, [3])
        if label is not None:
            label = torch.flip(label, [3])
    return image, label


class BaseDataset(data.Dataset):  # 训练、验证和测试使用不同的接口，见下
    def __init__(self, data_root, split):
        self.data_root = data_root

        file_list = osp.join('{}datalist'.format(settings.DATA_ROOT), split + '.txt')
        file_list = tuple(open(file_list, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_id = self.files[idx]
        return self._get_item(image_id)

    def _get_item(self, idx):
        raise NotImplemented


class TrainDataset(BaseDataset):
    def __init__(self, data_root=settings.DATA_ROOT, split='trainaug'):
        super(TrainDataset, self).__init__(data_root, split)

    def _get_item(self, image_id):
        image_path = osp.join(self.data_root,
                              'TrainVal_images/train_images',
                              image_id + '.jpg')
        label_path = osp.join(self.data_root,
                              'Trainval_parsing_annotations/train_segmentations',
                              image_id + '.png')

        image, label = fetch(image_path, label_path)
        image, label = scale(image, label)
        image, label = pad(image, label)
        image, label = crop(image, label)
        image, label = flip(image, label)

        return image[0], label[0, 0].long()
 

class ValDataset(BaseDataset):
    def __init__(self, data_root=settings.DATA_ROOT, split='val'):
        super(ValDataset, self).__init__(data_root, split)

    def _get_item(self, image_id):
        image_path = osp.join(self.data_root,
                              'TrainVal_images/val_images',
                              image_id + '.jpg')
        label_path = osp.join(self.data_root,
                              'Trainval_parsing_annotations/val_segmentations',
                              image_id + '.png')

        image, label = fetch(image_path, label_path)
        image, label = pad_inf(image, label)
        return image[0], label[0, 0].long(), image_id


class TestDataset(BaseDataset):
    def __init__(self, data_root=settings.DATA_ROOT, split='val'):
        super(TestDataset, self).__init__(data_root, split)

    def _get_item(self, image_id):
        image_path = osp.join(self.data_root,
                              'Testing_images',
                              image_id + '.jpg')

        image, label = fetch(image_path)
        image, label = pad_inf(image)
        return image[0], image_id


def test_dt():
    """
    测试
    """
    train_dt = TrainDataset()
    print('train', len(train_dt))
    for i in range(10):
        img, lbl = train_dt[i]
        print(img.shape, lbl.shape, img.mean(), np.unique(lbl))

    val_dt = ValDataset()
    print('val', len(val_dt))
    for i in range(10):
        img, lbl = val_dt[i]
        print(img.shape, lbl.shape, img.mean(), np.unique(lbl))


if __name__ == '__main__':
    test_dt()
