import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
from collections import OrderedDict

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x3 = self.layer3(x3)
        x3 = self.layer4(x3)

        return x1, x2, x3


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        source = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        new_dict = OrderedDict()
        for k1, v1 in model.state_dict().items():
            if k1 in source:
                new_dict[k1] = source[k1]
        model.load_state_dict(new_dict)
    return model

class ASPP2(nn.Module):
    def __init__(self, in_dim=2048, out_dim=256):
        super(ASPP2, self).__init__()

        self.pyramid1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
        self.pyramid2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
        self.pyramid3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
        self.pyramid4 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
        self.image_feature = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(out_dim * 5, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        feature1 = self.pyramid1(x)
        feature2 = self.pyramid2(x)
        feature3 = self.pyramid3(x)
        feature4 = self.pyramid4(x)
        image_feature = self.image_feature(x)
        image_feature = nn.functional.interpolate(image_feature, (x.shape[2], x.shape[3]), mode='bilinear')
        features = torch.cat((feature1, feature2, feature3, feature4, image_feature), dim=1)
        out = self.final(features)
        return out

class GuideBlock(nn.Module):
    def __init__(self):
        super(GuideBlock, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, dilation=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        self.weight1 = nn.Sequential(
            nn.Conv2d(64 + 64 + 5, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        
        self.side1 = nn.Conv2d(64, 64, 3, 1, 1)

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=2, padding=1, dilation=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        
        self.weight2 = nn.Sequential(
            nn.Conv2d(256 + 256 + 5, 256, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        
        self.side2 = nn.Conv2d(256, 256, 3, 1, 1)

        self.down3 = nn.Sequential(
           nn.Conv2d(256, 256, 3, stride=2, padding=1, dilation=1),
           nn.InstanceNorm2d(256),
           nn.ReLU(True)
        )

        self.weight3 = nn.Sequential(
            nn.Conv2d(256 + 256 + 5, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),            
        )

        self.side3 = nn.Conv2d(256, 256, 3, 1, 1)

    def forward(self, img, trimap, dist, feature1, feature2, feature3):
        x = self.down1(img)
        t1 = F.interpolate(trimap, (x.shape[2], x.shape[3]), mode='bilinear')
        d1 = F.interpolate(dist, (x.shape[2], x.shape[3]), mode='bilinear')
        x = torch.cat([x, feature1, t1, d1], dim=1)
        x = self.weight1(x)
        x1 = self.side1(x)
        x = self.down2(x)
        t2 = F.interpolate(trimap, (x.shape[2], x.shape[3]), mode='bilinear')
        d2 = F.interpolate(dist, (x.shape[2], x.shape[3]), mode='bilinear')
        x = torch.cat([x, feature2, t2, d2], dim=1)
        x = self.weight2(x)
        x2 = self.side2(x)
        x = self.down3(x)
        t3 = F.interpolate(trimap, (x.shape[2], x.shape[3]), mode='bilinear')
        d3 = F.interpolate(dist, (x.shape[2], x.shape[3]), mode='bilinear')
        x = torch.cat([x, feature3, t3, d3], dim=1)
        x = self.weight3(x)
        x3 = self.side3(x) 
        
        return x1, x2, x3

        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.skip1 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feature0, feature1, feature2, feature3):
        h = nn.functional.interpolate(feature3, (feature2.shape[2], feature2.shape[3]), mode='bilinear')
        feature2 = self.skip1(feature2)
        h = torch.cat((h, feature2), 1)
        h = self.decoder1(h)

        h = nn.functional.interpolate(h, (feature1.shape[2], feature1.shape[3]), mode='bilinear')
        feature1 = self.skip2(feature1)
        h = torch.cat((h, feature1), 1)
        h = self.decoder2(h)

        h = nn.functional.interpolate(h, (feature0.shape[2], feature0.shape[3]), mode='bilinear')
        h = torch.cat((h, feature0), 1)
        h = self.decoder3(h)
        prob = self.decoder4(h)

        return [h, prob]

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.fusion1 = nn.Sequential(
            nn.Conv2d(192, 64, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True))
        self.fusion2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True))
        self.fusion3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True))
        self.fusion4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True))
        self.fusion_out = nn.Conv2d(64, 1, 3, 1, 1)
        
    def forward(self, feature0, fg, bg):
        x = torch.cat([feature0, fg, bg], dim=1)
        x = self.fusion1(x)
        x = self.fusion2(x)
        x = self.fusion3(x)
        x = self.fusion4(x)
        x1 = F.sigmoid(self.fusion_out(x))
        return x1
        
class InteractNet(nn.Module):
    def __init__(self):
        super(InteractNet, self).__init__()
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.encoder = resnet50(pretrained=True)
        self.aspp = ASPP2(2048, 256)
        self.guide = GuideBlock()
        self.fnet = Decoder()
        self.bnet = Decoder()
        self.fusion = Fusion()


    def forward(self, img, trimap, dist_f, dist_b):
        dist = torch.cat([dist_f, dist_b], dim=1)
        feature0 = self.rgb_conv(img)
        feature1_, feature2_, feature3_ = self.encoder(img)
        feature3_ = self.aspp(feature3_)
        
        x1, x2, x3 = self.guide(img, trimap, dist, feature1_, feature2_, feature3_)
        feature1 = feature1_ + x1
        feature2 = feature2_ + x2
        feature3 = feature3_ + x3

        output_F = self.fnet(feature0, feature1, feature2, feature3)
        output_B = self.bnet(feature0, feature1, feature2, feature3)
        weight = self.fusion(feature0, output_F[0], output_B[0])
        alpha = output_F[1]*weight + (1-output_B[1])*(1-weight)
        return output_F, output_B, alpha
        
