import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

class SELayer(nn.Module):
    def __init__(self, channel, reduction=20):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)         #自适应的全局平均池化层
        self.fc = nn.Sequential(                        #学习通道权重的全连接网络fc
            nn.Linear(channel, channel+reduction),      #包括两个线性层和一个ReLU激活函数
            nn.ReLU(inplace=True),
            nn.Linear(channel+reduction, channel),
            nn.Sigmoid()                                #sig激活函数将权重限制在0到1之间，并将得到的权重应用在输入特征上，以得到加权后的特征输出
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        #self.add_module('norm1', nn.BatchNorm2d(num_input_features))    #批标准化层 32
        #self.add_module('relu1', nn.ReLU(inplace=True))                 #ReLU激活函数
        #self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,   #特征数量为2，对应信息输入输出 32,4*12
        #                                   kernel_size=1, stride=1, bias=False))    #1*1 卷积层，输出通道数bn_size * growth_rate

        #self.add_module('norm3', nn.BatchNorm2d(bn_size * growth_rate))  # 批标准化层 32
        #self.add_module('relu3', nn.ReLU(inplace=True))  # ReLU激活函数
        #self.add_module('conv3', nn.Conv2d(bn_size * growth_rate, (bn_size*2) * growth_rate,  # 特征数量为2，对应信息输入输出 32,4*12
         #                                  kernel_size=3, stride=1, padding=1,
          #                                 bias=False))  # 3*3 卷积层，输出通道数bn_size * growth_rate

   #     self.add_module('norm2', nn.BatchNorm2d((bn_size*2) * growth_rate)) #4*2*12
    #    self.add_module('relu2', nn.ReLU(inplace=True))
     #   self.add_module('conv2', nn.Conv2d((bn_size*2) * growth_rate, growth_rate,          #4*2*12,12
      #                                     kernel_size=3, stride=1, padding=1,      #3*3 卷积层
       #                                    bias=False))

        self.add_module('norm1', nn.BatchNorm2d(num_input_features))  # 批标准化层 32
        self.add_module('relu1', nn.ReLU(inplace=True))  # ReLU激活函数
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,  # 特征数量为2，对应信息输入输出 32,4*12
                                           kernel_size=1, stride=1, bias=False))  # 1*1 卷积层，输出通道数bn_size * growth_rate
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))  # 4*12
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,  # 4*12,12
                                           kernel_size=3, stride=1, padding=1,  # 3*3 卷积层
                                          bias=False))

        self.drop_rate = drop_rate
        self.se = SELayer(growth_rate)      #每个密集连接层都包含一个SE注意力模块

    def forward(self, input):
        new_features = super(_DenseLayer, self).forward(input)
        new_features = self.se(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,     #如果drop_rate大于0，则使用F.dropout方法对新特征new_features进行Dropout操作，防止过拟合
                                     training=self.training)
        return torch.cat([input, new_features], 1)                      #使用 torch.cat 将原始输入特征 input 和新特征 new_features 进行连接，拼接成最终的输出特征


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class DenseNetUnit(nn.Sequential):
    def __init__(self, channels, nb_flows, layers=5, growth_rate=12, num_init_features=32, bn_size=4, drop_rate=0.35):
        super(DenseNetUnit, self).__init__()

        if channels > 0:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(channels, num_init_features, kernel_size=3, padding=1)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True))
            ]))

            # Dense Block
            num_features = num_init_features
            num_layers = layers
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock', block)
            num_features = num_features + num_layers * growth_rate

            # Final batch norm
            self.features.add_module('normlast', nn.BatchNorm2d(num_features))
            self.features.add_module('convlast', nn.Conv2d(num_features, nb_flows,
                                                           kernel_size=1, padding=0, bias=False))

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()
        else:
            pass

    def forward(self, x):
        out = self.features(x)
        return out


class DenseNet(nn.Module):
    def __init__(self, nb_flows, drop_rate, channels):
        super(DenseNet, self).__init__()
        self.drop_rate = drop_rate
        self.nb_flows = nb_flows
        self.channels = channels

        self.close_feature = DenseNetUnit(channels[0], nb_flows)
        self.period_feature = DenseNetUnit(channels[1], nb_flows)
        self.trend_feature = DenseNetUnit(channels[2], nb_flows)

    def forward(self, inputs):
        out = 0
        if self.channels[0] > 0:
            out += self.close_feature(inputs[0])
        if self.channels[1] > 0:
            out += self.period_feature(inputs[1])
        if self.channels[2] > 0:
            out += self.trend_feature(inputs[2])

        return torch.sigmoid(out)
