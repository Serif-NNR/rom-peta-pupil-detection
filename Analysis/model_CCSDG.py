from torch import nn
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch




__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
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

    def __init__(self, block, layers, num_classes=1000, ini = 3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(ini, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('models/res_unet/resnet34-333f7ec4.pth'))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model






















class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)

        #diffY = up_p.size()[2] - x_p.size()[2]
        #diffX = up_p.size()[3] - x_p.size()[3]

        #x_p = F.pad(x_p, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        cat_p = torch.cat([up_p[:, :, 0:x_p.size()[-2], 0:x_p.size()[-1]], x_p], dim=1)
        return self.bn(F.relu(cat_p))


class UNet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        fea = x
        output = self.up5(x)

        return output

    def close(self):
        for sf in self.sfs: sf.remove()















class Projector(nn.Module):
    def __init__(self, output_size=1024):
        super(Projector, self).__init__()
        self.conv = nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(131072, output_size)

    def forward(self, x_in):
        x = self.conv(x_in)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x


class UNetCCSDG(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=1, pretrained=False):
        super().__init__()
        cut, lr_cut = [8, 6]

        self.name = "CCSDG_41Rate_"

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        #first_layer = layers[0]
        first_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        other_layers = layers[1:]
        base_layers = nn.Sequential(*other_layers)
        self.first_layer = first_layer
        self.rn = base_layers

        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [1, 3, 4, 5]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

    def forward_first_layer(self, x, tau=0.1):
        x = self.first_layer(x)  # 8 64 256 256

        channel_prompt_onehot = torch.softmax(self.channel_prompt/tau, dim=0)
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)
        return f_content, f_style

    def forward(self, x, tau=0.1):
        x = self.first_layer(x)  # 8 64 256 256

        channel_prompt_onehot = torch.softmax(self.channel_prompt / tau, dim=0)
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)

        x = F.relu(self.rn(f_content))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        output = self.up5(x)

        return output

    def close(self):
        for sf in self.sfs: sf.remove()




class UNetCCSDG_S(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=1, pretrained=False):
        super().__init__()
        cut, lr_cut = [8, 6]

        self.name = "CCSDG_S_41Rate_"

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        #first_layer = layers[0]
        first_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        other_layers = layers[1:-2]
        base_layers = nn.Sequential(*other_layers)
        self.first_layer = first_layer
        self.rn = base_layers

        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [1, 3, 4]]
        self.up1 = UnetBlock(128, 128, 128)
        self.up2 = UnetBlock(128, 64, 128)
        self.up3 = UnetBlock(128, 64, 128)
        #self.up4 = UnetBlock(128, 64, 128)

        self.up5 = nn.ConvTranspose2d(128, self.num_classes, 2, stride=2)

    def forward_first_layer(self, x, tau=0.1):
        x = self.first_layer(x)  # 8 64 256 256

        channel_prompt_onehot = torch.softmax(self.channel_prompt/tau, dim=0)
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)
        return f_content, f_style

    def forward(self, x, tau=0.1):
        x = self.first_layer(x)  # 8 64 256 256

        channel_prompt_onehot = torch.softmax(self.channel_prompt / tau, dim=0)
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)

        x = F.relu(self.rn(f_content))
        #x = self.up1(x, self.sfs[3].features)
        x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = self.up3(x, self.sfs[0].features)
        output = self.up5(x)

        return output

    def close(self):
        for sf in self.sfs: sf.remove()