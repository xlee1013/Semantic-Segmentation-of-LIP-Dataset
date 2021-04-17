import torch
from torch import nn
import torch.nn.functional as F
from networks.ResNet import resnet18


class FCN32S_Resnet(nn.Module):
    def __init__(self, num_classes):
        super(FCN32S_Resnet, self).__init__()
        backbone = resnet18()

        self.features4 = nn.Sequential(backbone.conv1,
                                       backbone.bn1,
                                       backbone.relu,
                                       backbone.maxpool,
                                       backbone.layer1,
                                       backbone.layer2,
                                       backbone.layer3
                                       )
        self.features5 = backbone.layer4

        self.score_conv_4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=1)

        fc6 = nn.Conv2d(512, 4096, kernel_size=3)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

    def forward(self, x):
        x_size = x.size()
        pool4 = self.features4(x)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore4 = F.interpolate(score_fr, pool4.size()[2:], mode="bilinear", align_corners=True)

        score_conv_3 = self.score_conv_4(pool4)
        upscore_pool3 = score_conv_3 + upscore4
        upscore_pool3 = F.interpolate(upscore_pool3, x_size[2:], mode="bilinear", align_corners=True)

        return upscore_pool3.contiguous()


def main():
    num_classes = 20
    in_batch, inchannel, in_h, in_w = 4, 3, 256, 256
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = FCN32S_Resnet(num_classes)
    out = net(x)
    print(out.shape)


if __name__ == '__main__':
    main()