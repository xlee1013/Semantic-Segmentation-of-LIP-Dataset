import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

from dataset import TestDataset
from network import Seg_FCN_Net
import settings
import utils

logger = settings.logger


class Session:
    def __init__(self, dt_split):

        self.log_dir = settings.LOG_DIR
        self.model_dir = settings.MODEL_DIR

        self.net = Seg_FCN_Net(settings.N_CLASSES).cuda()
        dataset = TestDataset(split=dt_split)  # 在这里 batchsize 只能设置为 1，因为每张图的尺寸不一致
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
        self.hist = 0

    def load_checkpoints(self, name):
        ckp_path = osp.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path, )
            logger.info('Load checkpoint %s.' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!' % ckp_path)
            return

        self.net.load_state_dict(obj['net'])

    def inf_batch(self, image, image_id):
        image = image.cuda()
        with torch.no_grad():
            logit = self.net(image)

        pred = logit.max(dim=1)[1]
        self.save_pred(pred, image_id, SAVE=True)

    def save_pred(self, pred, image_id, SAVE=True):
        if SAVE:
            mask = pred.cpu().squeeze(0).numpy()
            utils.save_pred_pic(mask, image_id[0], mode="test", show=False)


def main(ckp_name='latest.pth'):  # ckp_name 是已经保存模型的名字，载入之后使用该模型进行测试
    sess = Session(dt_split='test_id')
    sess.load_checkpoints(ckp_name)
    dt_iter = sess.dataloader
    sess.net.eval()

    for i, [image, image_id] in enumerate(dt_iter):
        sess.inf_batch(image, image_id)
        logger.info('{}.png has been saved.'.format(image_id[0]))


if __name__ == '__main__':
    main()
