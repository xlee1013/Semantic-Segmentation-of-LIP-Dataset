
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from dataset import TrainDataset
from network import Seg_FCN_Net
import settings
import utils

logger = settings.logger


def get_params(model, key):
    """
    返回卷积层、bn层、偏置，为了设置不同的学习率
    """
    if key == '1x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                yield m[1].weight

    if key == '1y':
        for m in model.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                yield m[1].weight

    if key == '2x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                if m[1].bias is not None:
                    yield m[1].bias


def poly_lr_scheduler(opt, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    opt.param_groups[0]['lr'] = 1 * new_lr
    opt.param_groups[1]['lr'] = 2 * new_lr


class Session:
    def __init__(self, dt_split):
        torch.manual_seed(66)

        self.log_dir = settings.LOG_DIR
        self.model_dir = settings.MODEL_DIR
        utils.ensure_dir(self.log_dir)
        utils.ensure_dir(self.model_dir)
        logger.info('Backbone:\t %s' % settings.BACKBONE)
        logger.info('Model:\t FCN%ds' % settings.STRIDE)
        logger.info('Number of classes:\t %d' % settings.N_CLASSES)
        logger.info('Batchzie:\t %d' % settings.BATCH_SIZE)
        logger.info('Training......')

        self.step = 1
        dataset = TrainDataset(split=dt_split)
        self.dataloader = DataLoader(  # 加载 batch 数据
            dataset, batch_size=settings.BATCH_SIZE, pin_memory=True,
            shuffle=True, drop_last=True)

        self.net = Seg_FCN_Net(settings.N_CLASSES).to(settings.DEVICE)  # 初始化模型
        self.opt = SGD(  # SGD优化器
            params=[  # 为卷积层、bn层和偏置项设置不同的学习率和衰减率
                {
                    'params': get_params(self.net, key='1x'),
                    'lr': 1 * settings.LR,
                    'weight_decay': settings.WEIGHT_DECAY,
                },
                {
                    'params': get_params(self.net, key='1y'),
                    'lr': 1 * settings.LR,
                    'weight_decay': settings.WEIGHT_DECAY,
                },
                {
                    'params': get_params(self.net, key='2x'),
                    'lr': 2 * settings.LR,
                    'weight_decay': 0.0,
                }],
            momentum=settings.LR_MOM)

    def write(self, out):  # 将训练信息载入日志
        out['lr'] = self.opt.param_groups[0]['lr']
        out['step'] = self.step

        utils.write_train_log_txt_file(out)  # 如果不会用 Tensorboard，可以在 logdir/train.txt查看loss损失，后续可以作图
        outputs = [
            '{}: {:.4g}'.format(k, v) 
            for k, v in out.items()]
        logger.info(' '.join(outputs))

    def save_checkpoints(self, name):  # 保存模型参数
        ckp_path = osp.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'step': self.step,
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name):  # 加载参数，只要给定已经保存了模型的名字，训练就可以随停随启
        ckp_path = osp.join(self.model_dir, name)
        try:
            if settings.CPU_OR_GPU == "cpu":  # 使用 cpu训练
                obj = torch.load(ckp_path)
            else:  # 使用自带显卡训练，速度更快
                obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.error('No checkpoint %s!' % ckp_path)
            return

        self.net.load_state_dict(obj['net'])
        self.step = obj['step']

    def train_batch(self, image, label):  # 训练一个 batch 的数据
        loss = self.net(image, label)

        loss = loss.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()


def main(ckp_name='latest.pth'):  # ckp_name 是已经保存模型的名字，载入之后可以接着训练
    sess = Session(dt_split='train_id')
    sess.load_checkpoints(ckp_name)

    dt_iter = iter(sess.dataloader)
    sess.net.train()

    while sess.step <= settings.ITER_MAX:  # 整个训练过程
        poly_lr_scheduler(
            opt=sess.opt,
            init_lr=settings.LR,
            iter=sess.step,
            lr_decay_iter=settings.LR_DECAY,
            max_iter=settings.ITER_MAX,
            power=settings.POLY_POWER)

        try:
            image, label = next(dt_iter)  # 提取 1个 batch的数据
        except StopIteration:
            dt_iter = iter(sess.dataloader)
            image, label = next(dt_iter)
        image, label = image.to(settings.DEVICE), label.to(settings.DEVICE)
        loss = sess.train_batch(image, label)
        out = {'loss': loss}
        sess.write(out)

        if sess.step % settings.ITER_SAVE == 0:  # 保存模型
            sess.save_checkpoints('step_%d.pth' % sess.step)
        if sess.step % settings.ITER_SAVE_LATSET == 0:
            sess.save_checkpoints('latest.pth')
        sess.step += 1

    sess.save_checkpoints('final.pth')


if __name__ == '__main__':
    main()
