import torch
from torch import nn
from torchvision import models

from utils import get_upsampling_weight
import settings


class FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(FCN32s, self).__init__()
        vgg = models.vgg16()
        vgg.load_state_dict(torch.load(settings.VGG16_MODEL_PATH))
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features5 = nn.Sequential(*features)

        fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)
        self.upscore.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 64))

    def forward(self, x):
        x_size = x.size()
        pool5 = self.features5(x)
        score_fr = self.score_fr(pool5)
        upscore = self.upscore(score_fr)

        return upscore[:, :, 19: (19 + x_size[2]), 19: (19 + x_size[3])].contiguous()


def main():
    num_classes = 20
    in_batch, inchannel, in_h, in_w = 4, 3, 256, 256
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = FCN32s(num_classes)
    out = net(x)
    print(out.shape)

if __name__ == '__main__':
    main()

