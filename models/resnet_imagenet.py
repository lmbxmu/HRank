
import torch.nn as nn


norm_mean, norm_var = 1.0, 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cp_rate=[0.], tmp_name=None):
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1.cp_rate = cp_rate[0]
        self.conv1.tmp_name = tmp_name
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2.cp_rate = cp_rate[1]
        self.conv2.tmp_name = tmp_name
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.conv3.cp_rate = cp_rate[2]
        self.conv3.tmp_name = tmp_name

        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class Downsample(nn.Module):
    def __init__(self, downsample):
        super(Downsample, self).__init__()
        self.downsample = downsample

    def forward(self, x):
        out = self.downsample(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, covcfg=None, compress_rate=None):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.covcfg = covcfg
        self.compress_rate = compress_rate
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.cp_rate = compress_rate[0]
        self.conv1.tmp_name = 'conv1'
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       cp_rate=compress_rate[1:3*num_blocks[0]+2],
                                       tmp_name='layer1')
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       cp_rate=compress_rate[3*num_blocks[0]+2:3*num_blocks[0]+3*num_blocks[1]+3],
                                       tmp_name='layer2')
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       cp_rate=compress_rate[3*num_blocks[0]+3*num_blocks[1]+3:3*num_blocks[0]+3*num_blocks[1]+3*num_blocks[2]+4],
                                       tmp_name='layer3')
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       cp_rate=compress_rate[3*num_blocks[0]+3*num_blocks[1]+3*num_blocks[2]+4:],
                                       tmp_name='layer4')

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, cp_rate, tmp_name):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv_short = nn.Conv2d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride, bias=False)
            conv_short.cp_rate = cp_rate[0]
            conv_short.tmp_name = tmp_name + '_shortcut'
            downsample = nn.Sequential(
                conv_short,
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, cp_rate=cp_rate[1:4],
                            tmp_name=tmp_name + '_block' + str(1)))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cp_rate=cp_rate[3 * i + 1:3 * i + 4],
                                tmp_name=tmp_name + '_block' + str(i + 1)))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # 256 x 56 x 56
        x = self.layer2(x)

        # 512 x 28 x 28
        x = self.layer3(x)

        # 1024 x 14 x 14
        x = self.layer4(x)

        # 2048 x 7 x 7
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_50(compress_rate=None):
    cov_cfg = [(3*i + 3) for i in range(3*3 + 1 + 4*3 + 1 + 6*3 + 1 + 3*3 + 1 + 1)]
    model = ResNet(ResBottleneck, [3, 4, 6, 3], covcfg=cov_cfg, compress_rate=compress_rate)
    return model
