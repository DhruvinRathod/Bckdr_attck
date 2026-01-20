# model/iresnet_old.py
# Fixed Paddle ArcFace-style iResNet (112x112 -> 7x7)

import paddle
import paddle.nn as nn

class IBasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2D(inplanes)
        self.conv1 = nn.Conv2D(inplanes, planes, 3, stride, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2D(planes, planes, 3, 1, 1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2D(inplanes, planes, 1, stride, bias_attr=False),
                nn.BatchNorm2D(planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return out + identity


class IResNet(nn.Layer):
    def __init__(self, layers, num_features=512, num_classes=1000, dropout=0.4):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2D(3, 64, 3, 1, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.prelu = nn.PReLU(64)

        # ✅ FIX: downsample at layer1 too (ArcFace standard)
        self.layer1 = self._make_layer(64,  layers[0], stride=2)  # 112 -> 56
        self.layer2 = self._make_layer(128, layers[1], stride=2)  # 56 -> 28
        self.layer3 = self._make_layer(256, layers[2], stride=2)  # 28 -> 14
        self.layer4 = self._make_layer(512, layers[3], stride=2)  # 14 -> 7

        self.bn2 = nn.BatchNorm2D(512)
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

        # ✅ FIX: final feature map is 7x7 for 112x112 input
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.bn3 = nn.BatchNorm1D(num_features)

        self.classifier = nn.Linear(num_features, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [IBasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = self.flatten(x)

        feat = self.bn3(self.fc(x))
        logits = self.classifier(feat)
        return feat, logits


def iresnet50(num_features=512, num_classes=1000, dropout=0.4):
    return IResNet([3, 4, 14, 3], num_features=num_features, num_classes=num_classes, dropout=dropout)
