import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ValDataset 
from metric import fast_hist, cal_scores
from network import Seg_FCN_Net
import settings
import utils

logger = settings.logger


class Session:
    def __init__(self, dt_split):

        self.log_dir = settings.LOG_DIR
        self.model_dir = settings.MODEL_DIR

        self.net = Seg_FCN_Net(settings.N_CLASSES).cuda()
        dataset = ValDataset(split=dt_split)  # 在这里 batchsize 只能设置为 1，因为每张图的尺寸不一致
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

    def inf_batch(self, image, label, image_id):
        image = image.cuda()
        label = label.cuda()
        with torch.no_grad():
            logit = self.net(image)

        pred = logit.max(dim=1)[1]
        self.save_pred(pred, image_id, SAVE=False)  # 如果SAVE设置为 True，则在验证阶段保存分割结果图
        self.hist += fast_hist(label, pred)

    def save_pred(self, pred, image_id, SAVE=False):
        if SAVE:
            mask = pred.cpu().squeeze(0).numpy()
            utils.save_pred_pic(mask, image_id[0], mode="eval", show=False)


def main(ckp_name='latest.pth'):  # ckp_name 是已经保存模型的名字，载入之后可以使用该模型进行验证
    sess = Session(dt_split='val_id')
    sess.load_checkpoints(ckp_name)
    dt_iter = sess.dataloader
    sess.net.eval()

    for i, [image, label, image_id] in enumerate(dt_iter):
        sess.inf_batch(image, label, image_id)
        if i % 10 == 0:
            logger.info('num-%d' % i)
            scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
            for k, v in scores.items():
                logger.info('%s-%f' % (k, v))

    scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
    for k, v in scores.items():
        logger.info('%s-%f' % (k, v))
    logger.info('')
    for k, v in cls_iu.items():
        logger.info('%s-%f' % (k, v))


if __name__ == '__main__':
    main()
