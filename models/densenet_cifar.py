import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np


norm_mean, norm_var = 0.0, 1.0

cov_cfg=[(3*i+1) for i in range(12*3+2+1)]


class DenseBasicBlock(nn.Module):
    def __init__(self, inplanes, filters, index, expansion=1, growthRate=12, dropRate=0, compress_rate=0., tmp_name=None):
        super(DenseBasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(filters, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.conv1.cp_rate = compress_rate
        self.conv1.tmp_name = tmp_name

        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, filters, index, compress_rate, tmp_name, last_prune_num):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(filters, outplanes, kernel_size=1,
                               bias=False)
        self.conv1.cp_rate = compress_rate
        self.conv1.tmp_name = tmp_name
        self.conv1.last_prune_num=last_prune_num

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, depth=40, block=DenseBasicBlock,
        dropRate=0, num_classes=10, growthRate=12, compressionRate=1, filters=None, indexes=None,compress_rate=None):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if 'DenseBasicBlock' in str(block) else (depth - 4) // 6

        transition = Transition
        if filters == None:
            filters = []
            start = growthRate*2
            for i in range(3):
                filters.append([start + growthRate*i for i in range(n+1)])
                start = (start + growthRate*n) // compressionRate
            filters = [item for sub_list in filters for item in sub_list]

            indexes = []
            for f in filters:
                indexes.append(np.arange(f))

        self.covcfg=cov_cfg
        self.compress_rate=compress_rate

        self.growthRate = growthRate
        self.dropRate = dropRate

        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.conv1.cp_rate=compress_rate[0]
        self.conv1.tmp_name = 'conv1'
        self.last_prune_num=self.inplanes*compress_rate[0]

        self.dense1 = self._make_denseblock(block, n, filters[0:n], indexes[0:n], compress_rate[1:n+1],'dense1', self.last_prune_num)
        self.trans1 = self._make_transition(transition, compressionRate, filters[n], indexes[n], compress_rate[n+1],'trans1', self.last_prune_num)
        self.dense2 = self._make_denseblock(block, n, filters[n+1:2*n+1], indexes[n+1:2*n+1], compress_rate[n+2:2*n+2],'dense2', self.last_prune_num)
        self.trans2 = self._make_transition(transition, compressionRate, filters[2*n+1], indexes[2*n+1], compress_rate[2*n+2],'trans2', self.last_prune_num)
        self.dense3 = self._make_denseblock(block, n, filters[2*n+2:3*n+2], indexes[2*n+2:3*n+2], compress_rate[2*n+3:3*n+3],'dense3', self.last_prune_num)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, filters, indexes, compress_rate, tmp_name, last_prune_num):
        layers = []
        assert blocks == len(filters), 'Length of the filters parameter is not right.'
        assert blocks == len(indexes), 'Length of the indexes parameter is not right.'
        for i in range(blocks):
            self.last_prune_num+=int(compress_rate[i]*self.growthRate)
            layers.append(block(self.inplanes, filters=filters[i], index=indexes[i],
                                growthRate=self.growthRate, dropRate=self.dropRate, compress_rate=compress_rate[i], tmp_name=tmp_name+'_'+str(i)))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, transition, compressionRate, filters, index, compress_rate, tmp_name, last_prune_num):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        self.last_prune_num=int(compress_rate*filters)
        return transition(inplanes, outplanes, filters, index, compress_rate, tmp_name, last_prune_num)

    def forward(self, x):
        x = self.conv1(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def densenet_40(compress_rate=None):
    return DenseNet(depth=40, block=DenseBasicBlock, compressionRate=1, compress_rate=compress_rate)
