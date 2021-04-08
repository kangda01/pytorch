from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )



class DWandPWResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(DWandPWResidual, self).__init__()
        # hidden_channel = in_channel
        # self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []

        # 3x3 depthwise conv
        layers.append(ConvBNReLU(in_channel, in_channel,stride=stride, groups=in_channel))
        # 1x1 pointwise conv
        layers.append(ConvBNReLU(in_channel, out_channel, kernel_size=1))
        layers.append(nn.BatchNorm2d(out_channel))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)






class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV1, self).__init__()
        block = DWandPWResidual
        input_channel = 32
        last_channel = 1024
        # input_channel = _make_divisible(32 * alpha, round_nearest)
        # last_channel = _make_divisible(1280 * alpha, round_nearest)

        PWandDW_residual_setting = [
            # n, c,  s
            [1, 64,  1],
            [1, 128,  2],
            [1, 128,  1],
            [1, 256,  2],
            [1, 256,  1],
            [1, 512,  2],
            [5, 512,  1],
            [1, 1024, 2],
            [1, 1024, 2],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # features.extend([block()])
        # building inverted residual residual blockes
        for n, c, s in PWandDW_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride))
                input_channel = output_channel

        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x






# from torchstat import stat
#
#
# model = MobileNetV2()
# stat(model, (3, 224, 224))
