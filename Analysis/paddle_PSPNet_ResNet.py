import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import activation
import os
from torchvision import transforms as T



def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if torch.cuda.is_available() == True:
        return nn.BatchNorm2d(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


class SeparableConvBNReLU(nn.Module):
    """Depthwise Separable Convolution."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: str = 'same',
                 **kwargs: dict):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            **kwargs)
        self.piontwise_conv = ConvBNReLU(
            in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class ConvBN(nn.Module):
    """Basic conv bn layer"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: str = 'same',
                 **kwargs: dict):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._batch_norm = SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvBNReLU(nn.Module):
    """Basic conv bn relu layer."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: str = 'same',
                 **kwargs: dict):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._batch_norm = SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        #x = self._batch_norm(x)
        x = F.relu(x)
        return x


class Activation(nn.Module):
    """
    The wrapper of activations.
    Args:
        act (str, optional): The activation name in lowercase. It must be one of ['elu', 'gelu',
            'hardshrink', 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid',
            'softmax', 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax',
            'hsigmoid']. Default: None, means identical transformation.
    Returns:
        A callable object of Activation.
    Raises:
        KeyError: When parameter `act` is not in the optional range.
    Examples:
        from paddleseg.models.common.activation import Activation
        relu = Activation("relu")
        print(relu)
        # <class 'paddle.nn.layer.activation.ReLU'>
        sigmoid = Activation("sigmoid")
        print(sigmoid)
        # <class 'paddle.nn.layer.activation.Sigmoid'>
        not_exit_one = Activation("not_exit_one")
        # KeyError: "not_exit_one does not exist in the current dict_keys(['elu', 'gelu', 'hardshrink',
        # 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid', 'softmax',
        # 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax', 'hsigmoid'])"
    """

    def __init__(self, act: str = None):
        super().__init__()

        self._act = act
        upper_act_names = activation.__dict__.keys()
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))

        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                self.act_func = eval("activation.{}()".format(act_name))
            else:
                raise KeyError("{} does not exist in the current {}".format(
                    act, act_dict.keys()))

    def forward(self, x):

        if self._act is not None:
            return self.act_func(x)
        else:
            return x


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        use_sep_conv (bool, optional): If using separable conv in ASPP module. Default: False.
        image_pooling (bool, optional): If augmented with image-level features. Default: False
    """

    def __init__(self,
                 aspp_ratios: tuple,
                 in_channels: int,
                 out_channels: int,
                 align_corners: bool,
                 use_sep_conv: bool = False,
                 image_pooling: bool = False):
        super().__init__()

        self.align_corners = align_corners
        self.aspp_blocks = nn.LayerList()

        for ratio in aspp_ratios:
            if use_sep_conv and ratio > 1:
                conv_func = SeparableConvBNReLU
            else:
                conv_func = ConvBNReLU

            block = conv_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if ratio == 1 else 3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio)
            self.aspp_blocks.append(block)

        out_size = len(self.aspp_blocks)

        if image_pooling:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2D(output_size=(1, 1)),
                ConvBNReLU(in_channels, out_channels, kernel_size=1, bias_attr=False))
            out_size += 1
        self.image_pooling = image_pooling

        self.conv_bn_relu = ConvBNReLU(
            in_channels=out_channels * out_size,
            out_channels=out_channels,
            kernel_size=1)

        self.dropout = nn.Dropout(p=0.1)  # drop rate

    def forward(self, x):
        outputs = []
        for block in self.aspp_blocks:
            y = block(x)
            y = F.interpolate(
                y,
                x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            outputs.append(y)

        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            img_avg = F.interpolate(
                img_avg,
                x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            outputs.append(img_avg)

        x = np.concat(outputs, axis=1)
        x = self.conv_bn_relu(x)
        x = self.dropout(x)

        return x


class AuxLayer(nn.Module):
    """
    The auxiliary layer implementation for auxiliary loss.
    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self,
                 in_channels: int,
                 inter_channels: int,
                 out_channels: int,
                 dropout_prob: float = 0.1,
                 **kwargs):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            **kwargs)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2D(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, name: str = None):
        return torch.add(x, y)

class PPModule(nn.Module):
    """
    Pyramid pooling module originally in PSPNet.
    Args:
        in_channels (int): The number of intput channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 2, 3, 6).
        dim_reduction (bool, optional): A bool value represents if reducing dimension after pooling. Default: True.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bin_sizes,
                 dim_reduction,
                 align_corners,
                 device):
        super().__init__()

        self.bin_sizes = bin_sizes

        inter_channels = in_channels
        if dim_reduction:
            inter_channels = in_channels // len(bin_sizes)

        # we use dimension reduction after pooling mentioned in original implementation.
        self.stages = [
            self._make_stage(in_channels, inter_channels, size, device)
            for size in bin_sizes
        ]


        self.conv_bn_relu2 = ConvBNReLU(
            in_channels=in_channels + inter_channels * len(bin_sizes),
            out_channels=out_channels,
            kernel_size=3,
            padding=1)

        self.align_corners = align_corners

    def _make_stage(self, in_channels: int, out_channels: int, size: int, device):
        """
        Create one pooling layer.
        In our implementation, we adopt the same dimension reduction as the original paper that might be
        slightly different with other implementations.
        After pooling, the channels are reduced to 1/len(bin_sizes) immediately, while some other implementations
        keep the channels to be same.
        Args:
            in_channels (int): The number of intput channels to pyramid pooling module.
            size (int): The out size of the pooled layer.
        Returns:
            conv (Tensor): A tensor after Pyramid Pooling Module.
        """

        prior = nn.AdaptiveAvgPool2d(output_size=(size, size)).to(device)
        conv = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1).to(device)

        return nn.Sequential(prior, conv)

    def forward(self, input):
        cat_layers = []
        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                input.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            cat_layers.append(x)
        cat_layers = [input] + cat_layers[::-1]
        cat = torch.cat(cat_layers, axis=1)
        out = self.conv_bn_relu2(cat)

        return out


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 is_vd_mode: bool = False,
                 act: str = None,
                 data_format: str = 'NCHW'):
        super().__init__()
        if dilation != 1 and kernel_size != 3:
            raise RuntimeError("When the dilation isn't 1," \
                               "the kernel_size should be 3.")

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
            ceil_mode=True)
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 \
                if dilation == 1 else dilation,
            dilation=dilation,
            groups=groups)

        self._batch_norm = SyncBatchNorm(
            out_channels)
        self._act_op = Activation(act=act)

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        y = self._act_op(y)

        return y


class BottleneckBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 shortcut: bool = True,
                 if_first: bool  = False,
                 dilation: int = 1,
                 data_format: str = 'NCHW'):
        super().__init__()

        self.data_format = data_format
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            data_format=data_format)

        self.dilation = dilation

        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            dilation=dilation,
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            data_format=data_format)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True,
                data_format=data_format)

        self.shortcut = shortcut
        # NOTE: Use the wrap layer for quantization training
        self.add = Add()
        self.relu = Activation(act="relu")

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = self.add(short, conv2)
        y = self.relu(y)
        return y


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 dilation: int = 1,
                 shortcut: bool = True,
                 if_first: bool = False,
                 data_format: str = 'NCHW'):
        super().__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            act='relu',
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=dilation,
            act=None,
            data_format=data_format)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True,
                data_format=data_format)

        self.shortcut = shortcut
        self.dilation = dilation
        self.data_format = data_format
        self.add = Add()
        self.relu = Activation(act="relu")

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = self.add(short, conv1)
        y = self.relu(y)

        return y


class ResNet_vd(nn.Module):
    """
    The ResNet_vd implementation based on PaddlePaddle.
    The original article refers to Jingdong
    Tong He, et, al. "Bag of Tricks for Image Classification with Convolutional Neural Networks"
    (https://arxiv.org/pdf/1812.01187.pdf).
    Args:
        layers (int, optional): The layers of ResNet_vd. The supported layers are (18, 34, 50, 101, 152, 200). Default: 50.
        output_stride (int, optional): The stride of output features compared to input images. It is 8 or 16. Default: 8.
        multi_grid (tuple|list, optional): The grid of stage4. Defult: (1, 1, 1).
        pretrained (str, optional): The path of pretrained model.
    """

    def __init__(self,
                 layers=18,
                 device = "",
                 output_stride=8,
                 multi_grid=(1, 1, 1),
                 pretrained=None,
                 data_format='NCHW'):
        super().__init__()

        self.data_format = data_format
        self.conv1_logit = None  # for gscnn shape stream
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024
                        ] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        # for channels of four returned stages
        self.feat_channels = [c * 4 for c in num_filters
                              ] if layers >= 50 else num_filters

        dilation_dict = None
        if output_stride == 8:
            dilation_dict = {2: 2, 3: 4}
        elif output_stride == 16:
            dilation_dict = {3: 2}

        self.conv1_1 = ConvBNLayer(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act='relu',
            data_format=data_format)
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            data_format=data_format)
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
            data_format=data_format)
        self.pool2d_max = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1)

        # self.block_list = []
        self.stage_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)

                    ###############################################################################
                    # Add dilation rate for some segmentation tasks, if dilation_dict is not None.
                    dilation_rate = dilation_dict[
                        block] if dilation_dict and block in dilation_dict else 1

                    # Actually block here is 'stage', and i is 'block' in 'stage'
                    # At the stage 4, expand the the dilation_rate if given multi_grid
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]
                    ###############################################################################

                    bottleneck_block = nn.Sequential(

                        BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0
                                        and dilation_rate == 1 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            dilation=dilation_rate,
                            data_format=data_format).to(device))

                    block_list.append(bottleneck_block)
                    shortcut = True
                self.stage_list.append(block_list)
        else:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    dilation_rate = dilation_dict[block] \
                        if dilation_dict and block in dilation_dict else 1
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]

                    basic_block = nn.Sequential(

                        BasicBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 \
                                        and dilation_rate == 1 else 1,
                            dilation=dilation_rate,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            data_format=data_format).to(device)).to(device)
                    block_list.append(basic_block)
                    shortcut = True
                self.stage_list.append(block_list)

        self.pretrained = pretrained

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        self.conv1_logit = y.clone()
        y = self.pool2d_max(y)

        # A feature list saves the output feature map of each stage.
        feat_list = []
        for stage in self.stage_list:
            for block in stage:
                y = block(y)
            feat_list.append(y)

        return feat_list


def ResNet50_vd(device, **args):
    model = ResNet_vd(layers=18, device=device, **args).to(device)
    return model























class PSPNet(nn.Module):
    """
    The PSPNet implementation based on PaddlePaddle.
    The original article refers to
    Zhao, Hengshuang, et al. "Pyramid scene parsing network"
    (https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf).
    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
        pp_out_channels (int, optional): The output channels after Pyramid Pooling Module. Default: 1024.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1,2,3,6).
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes=1,
                 backbone_indices=(2, 3),
                 device = "",
                 pp_out_channels=1024,
                 bin_sizes=(1, 2, 3, 6),
                 enable_auxiliary_loss=  False,
                 align_corners= False,
                 pretrained= None):
        super(PSPNet, self).__init__()
        self.name = "PSPNet_"
        self.backbone = ResNet50_vd(device)
        backbone_channels = [
            self.backbone.feat_channels[i] for i in backbone_indices
        ]
        self.sig = nn.Sigmoid()
        self.head = PSPNetHead(device, num_classes, backbone_indices, backbone_channels,
                               pp_out_channels, bin_sizes,
                               enable_auxiliary_loss, align_corners)
        self.align_corners = align_corners
        self.transforms = T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        if pretrained is not None:
            model_dict = torch.load(pretrained)
            self.set_dict(model_dict)
            print("load custom parameters success")

        '''else:
            checkpoint = os.path.join(self.directory, 'model.pdparams')
            model_dict = torch.load(checkpoint)
            self.set_dict(model_dict)
            print("load pretrained parameters success")
            '''

    #def transform(self, img):
    #    return self.transforms(img)

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        a = torch.reshape(x, x.shape[2:])
        b = [
            F.interpolate(
                logit,
                x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]
        c = self.sig(b[0])
        return c


class PSPNetHead(nn.Module):
    """
    The PSPNetHead implementation.
    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            The first index will be taken as a deep-supervision feature in auxiliary layer;
            the second one will be taken as input of Pyramid Pooling Module (PPModule).
            Usually backbone consists of four downsampling stage, and return an output of
            each stage. If we set it as (2, 3) in ResNet, that means taking feature map of the third
            stage (res4b22) in backbone, and feature map of the fourth stage (res5c) as input of PPModule.
        backbone_channels (tuple): The same length with "backbone_indices". It indicates the channels of corresponding index.
        pp_out_channels (int): The output channels after Pyramid Pooling Module.
        bin_sizes (tuple): The out size of pooled feature maps.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self, device, num_classes, backbone_indices, backbone_channels,
                 pp_out_channels, bin_sizes, enable_auxiliary_loss,
                 align_corners):

        super().__init__()

        self.backbone_indices = backbone_indices

        self.psp_module = PPModule(
            in_channels=backbone_channels[1],
            out_channels=pp_out_channels,
            bin_sizes=bin_sizes,
            dim_reduction=True,
            align_corners=align_corners,
        device=device)

        self.dropout = nn.Dropout(p=0.1)  # dropout_prob

        self.conv = nn.Conv2d(
            in_channels=pp_out_channels,
            out_channels=num_classes,
            kernel_size=1)

        if enable_auxiliary_loss and False:
            self.auxlayer = torch.AuxLayer(
                in_channels=backbone_channels[0],
                inter_channels=backbone_channels[0] // 4,
                out_channels=num_classes)

        self.enable_auxiliary_loss = enable_auxiliary_loss

    def forward(self, feat_list):
        logit_list = []
        i = self.backbone_indices[1]
        x = feat_list[i]
        x = self.psp_module(x)
        x = self.dropout(x)
        logit = self.conv(x)
        logit_list.append(logit)

        if self.enable_auxiliary_loss and False:
            auxiliary_feat = feat_list[self.backbone_indices[0]]
            auxiliary_logit = self.auxlayer(auxiliary_feat)
            logit_list.append(auxiliary_logit)

        return logit_list



if __name__ == "__main__":
    model = PSPNet()
    a = model(torch.randn(1, 1, 320, 240))
    print("Done")







