import torch
import torch.nn as nn
import torch.nn.functional as F

import settings


def FCN(n_classes):  # 定义网络类型，具体的网络可见 networks/ 路径下
    if settings.BACKBONE == "VGGNet":
        if settings.STRIDE == 8:
            from networks.FCN8s import FCN8s
            return FCN8s(n_classes)

        elif settings.STRIDE == 16:
            from networks.FCN16s import FCN16s
            return FCN16s(n_classes)

        elif settings.STRIDE == 32:
            from networks.FCN32s import FCN32s
            return FCN32s(n_classes)

    if settings.BACKBONE == "ResNet":
        if settings.STRIDE == 8:
            from networks.FCN8S_Resnet import FCN8S_Resnet
            return FCN8S_Resnet(n_classes)

        elif settings.STRIDE == 16:
            from networks.FCN16S_Resnet import FCN16S_Resnet
            return FCN16S_Resnet(n_classes)

        elif settings.STRIDE == 32:
            from networks.FCN32S_Resnet import FCN32S_Resnet
            return FCN32S_Resnet(n_classes)


class CrossEntropyLoss2d(nn.Module):  # 损失函数使用交叉熵损失函数
    def __init__(self, weight=None, reduction='none', ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction,
                                   ignore_index=ignore_index)

    def forward(self, inputs, targets):
        loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return loss.mean(dim=2).mean(dim=1)


class Seg_FCN_Net(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # FCN
        self.fcn = FCN(n_classes)

        # Loss
        self.crit = CrossEntropyLoss2d(ignore_index=settings.IGNORE_LABEL,
                                       reduction='none')

    def forward(self, img, lbl=None):
        pred = self.fcn(img)

        if self.training and lbl is not None:  # 训练时返回 loss
            loss = self.crit(pred, lbl)
            return loss
        else:  # 验证和测试时返回预测特征图，尺寸为[batchsize, 20, h, w]
            return pred


if __name__ == "__main__":
    # 测试网络
    net = Seg_FCN_Net(20)
    net.eval()
    img = torch.rand([1, 3, 256, 256])
    labal = None
    pred = net(img)
    print(pred.shape)
