import torch
from torch import nn
from torchvision import models
from utils import get_upsampling_weight
import settings


class FCN16s(nn.Module):
    def __init__(self, num_classes):
        super(FCN16s, self).__init__()
        vgg = models.vgg16()
        vgg.load_state_dict(torch.load(settings.VGG16_MODEL_PATH))
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features4 = nn.Sequential(*features[: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, bias=False)
        self.upscore2.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore16.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 32))

    def forward(self, x):
        x_size = x.size()
        pool4 = self.features4(x)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * pool4)
        upscore16 = self.upscore16(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                   + upscore2)
        return upscore16[:, :, 27: (27 + x_size[2]), 27: (27 + x_size[3])].contiguous()


def main():
    num_classes = 20
    in_batch, inchannel, in_h, in_w = 4, 3, 67, 55
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = FCN16s(num_classes)
    out = net(x)
    print(out.shape)


if __name__ == '__main__':
    main()