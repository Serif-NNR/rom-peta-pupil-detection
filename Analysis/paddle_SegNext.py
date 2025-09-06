
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import activation
import os
from torchvision import transforms as T
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def to_2tuple(x):
    return tuple([x] * 2)
































def SyncBatchNorm(*args, **kwargs):
        return nn.BatchNorm2d(*args, **kwargs).to(device)


class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class ConvGNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding="same",
                 num_groups=1,
                 act_type=None,
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding).to(device)

        if "data_format" in kwargs:
            data_format = kwargs["data_format"]
        else:
            data_format = "NCHW"
        self._group_norm = nn.GroupNorm(
            num_groups, out_channels).to(device)
        self._act_type = act_type
        if act_type is not None:
            self._act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        x = self._group_norm(x)
        if self._act_type is not None:
            x = self._act(x)
        return x


class ConvNormAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 act_type=None,
                 norm=None,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'

        self._norm = norm if norm is not None else None

        self._act_type = act_type
        if act_type is not None:
            self._act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        if self._norm is not None:
            x = self._norm(x)
        if self._act_type is not None:
            x = self._act(x)
        return x


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 act_type=None,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)

        self._act_type = act_type
        if act_type is not None:
            self._act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        if self._act_type is not None:
            x = self._act(x)
        return x


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvReLUPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)
        self._relu = nn.ReLU(inplace=True)
        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self._relu(x)
        x = self._max_pool(x)
        return x


class SeparableConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 pointwise_bias=None,
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self.piontwise_conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            data_format=data_format,
            bias_attr=pointwise_bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels)

    def forward(self, x):
        x = self.depthwise_conv(x)
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
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1,
                 **kwargs):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class JPU(nn.Module):
    """
    Joint Pyramid Upsampling of FCN.
    The original paper refers to
        Wu, Huikai, et al. "Fastfcn: Rethinking dilated convolution in the backbone for semantic segmentation." arXiv preprint arXiv:1903.11816 (2019).
    """

    def __init__(self, in_channels, width=512):
        super().__init__()

        self.conv5 = ConvBNReLU(
            in_channels[-1], width, 3, padding=1, bias_attr=False)
        self.conv4 = ConvBNReLU(
            in_channels[-2], width, 3, padding=1, bias_attr=False)
        self.conv3 = ConvBNReLU(
            in_channels[-3], width, 3, padding=1, bias_attr=False)

        self.dilation1 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=1,
            pointwise_bias=False,
            dilation=1,
            bias_attr=False,
            stride=1, )
        self.dilation2 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=2,
            pointwise_bias=False,
            dilation=2,
            bias_attr=False,
            stride=1)
        self.dilation3 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=4,
            pointwise_bias=False,
            dilation=4,
            bias_attr=False,
            stride=1)
        self.dilation4 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=8,
            pointwise_bias=False,
            dilation=8,
            bias_attr=False,
            stride=1)

    def forward(self, *inputs):
        feats = [
            self.conv5(inputs[-1]), self.conv4(inputs[-2]),
            self.conv3(inputs[-3])
        ]
        size = feats[-1].shape[2:]
        feats[-2] = F.interpolate(
            feats[-2], size, mode='bilinear', align_corners=True)
        feats[-3] = F.interpolate(
            feats[-3], size, mode='bilinear', align_corners=True)
        # SRO
        feat = torch.cat(feats, dim=1)
        feat = torch.cat(
            [
                self.dilation1(feat), self.dilation2(feat),
                self.dilation3(feat), self.dilation4(feat)
            ],
            dim=1)

        return inputs[0], inputs[1], inputs[2], feat


class ConvBNPReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._prelu = nn.PReLU()

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._prelu(x)
        return x


class ConvBNLeakyReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x

















































































from abc import abstractmethod, ABCMeta




class _MatrixDecomposition2DBase(nn.Module, metaclass=ABCMeta):
    """
    The base implementation of 2d matrix decomposition.
    The original article refers to
    Yuanduo Hong, Huihui Pan, Weichao Sun, et al. "Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes"
    (https://arxiv.org/abs/2101.06085)
    """

    def __init__(self, args=None):
        super().__init__()
        if args is None:
            args = dict()
        elif not isinstance(args, dict):
            raise TypeError("`args` must be a dict, but got {}".foramt(
                args.__class__.__name__))

        self.spatial = args.setdefault("SPATIAL", True)

        self.S = args.setdefault("MD_S", 1)
        self.D = args.setdefault("MD_D", 512)
        self.R = args.setdefault("MD_R", 64)

        self.train_steps = args.setdefault("TRAIN_STEPS", 6)
        self.eval_steps = args.setdefault("EVAL_STEPS", 7)

        self.inv_t = args.setdefault("INV_T", 100)
        self.eta = args.setdefault("ETA", 0.9)

        self.rand_init = args.setdefault("RAND_INIT", True)

    def _build_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def forward(self, x):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, "bases"):
            bases = self._build_bases(1, self.S, D, self.R).to(device)
            #self.register_buffer("bases", bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R).to(device)
        else:
            bases = torch.repeat_interleave(self.bases, B, 0).to(device)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        # bases = bases.reshape([B, self.S, D, self.R])

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))

        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator
        del numerator
        # multiplication update
        coef = coef  / (denominator + 1e-6)

        return coef



















def drop_path(x, drop_prob=0., training=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = torch.tensor(1 - drop_prob).to(device)
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype).to(device)
    random_tensor = torch.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


































def get_depthwise_conv(dim, kernel_size=3):
    if isinstance(kernel_size, int):
        kernel_size = to_2tuple(kernel_size)
    padding = tuple([k // 2 for k in kernel_size])
    return nn.Conv2d(
        dim, dim, kernel_size, padding=padding, groups=dim).to(device)


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1).to(device)
        self.dwconv = get_depthwise_conv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1).to(device)
        self.drop = nn.Dropout(drop).to(device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels, ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1),
            SyncBatchNorm(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1),
            SyncBatchNorm(out_channels)).to(device)

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        return x, H, W

from copy import deepcopy
class AttentionModule(nn.Module):
    """
    AttentionModule Layer, which contains some depth-wise strip convolutions.
    Args:
        dim (int): Number of input channels.
        kernel_sizes (list[int], optional): The height or width of each strip convolution kernel. Default: [7, 11, 21].
    """

    def __init__(self, dim, kernel_sizes=[7, 11, 21]):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.dwconvs = [
            nn.Sequential(get_depthwise_conv(dim, (1, k)),
                          get_depthwise_conv(dim, (k, 1)))
            for i, k in enumerate(kernel_sizes)
        ]

        self.conv_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        #u = torch.clone(x)
        attn = self.conv0(x)

        attns = [m(attn) for m in self.dwconvs]

        attn += sum(attns)
        # UPDATE
        attn = self.conv_out(attn)

        return attn * x


class SpatialAttention(nn.Module):
    """
    SpatialAttention Layer.
    Args:
        d_model (int): Number of input channels.
        atten_kernel_sizes (list[int], optional): The height or width of each strip convolution kernel in attention module.
            Default: [7, 11, 21].
    """

    def __init__(self, d_model, atten_kernel_sizes=[7, 11, 21]):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1).to(device)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model, atten_kernel_sizes).to(device)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1).to(device)

    def forward(self, x):
        shorcut = torch.clone(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    """
    MSCAN Block.
    Args:
        dim (int): Number of feature channels.
        atten_kernel_sizes (list[int], optional): The height or width of each strip convolution kernel in attention module.
            Default: [7, 11, 21].
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU.
    """

    def __init__(
            self,
            dim,
            atten_kernel_sizes=[7, 11, 21],
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU, ):
        super().__init__()
        self.norm1 = SyncBatchNorm(dim).to(device)
        self.attn = SpatialAttention(dim, atten_kernel_sizes).to(device)
        self.drop_path = DropPath(
            drop_path).to(device) if drop_path > 0.0 else nn.Identity().to(device)
        self.norm2 = SyncBatchNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)

        layer_scale_init_value = torch.full(
            [dim, 1, 1], fill_value=1e-2, dtype=torch.float32)
        self.layer_scale_1 = torch.rand(
            dim, 1, 1, dtype=torch.float32).to(device)
        self.layer_scale_2 = torch.rand(
            dim, 1, 1, dtype=torch.float32).to(device)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        x = x.view(B, C, N)
        x = x.transpose(1, 2)
        return x


class OverlapPatchEmbed(nn.Module):
    """
    An Opverlaping Image to Patch Embedding Layer.
    Args:
        patch_size (int, optional): Patch token size. Default: 7.
        stride (int, optional): Stride of Convolution in OverlapPatchEmbed. Default: 4.
        in_chans (int, optional): Number of input image channels. Default: 3.
        embed_dim (int, optional): Number of linear projection output channels. Default: 768.
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)).to(device)
        self.norm = SyncBatchNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2:]
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)
        return x, H, W


def _check_length(*args):
    target_length = len(args[0])
    for item in args:
        if target_length != len(item):
            return False
    return True



class MSCAN(nn.Module):
    """
    The MSCAN implementation based on PaddlePaddle.
    The original article refers to
    Guo, Meng-Hao, et al. "SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation"
    (https://arxiv.org/pdf/2209.08575.pdf)
    Args:
        in_channels (int, optional): Number of input image channels. Default: 3.
        embed_dims (list[int], optional): Number of each stage output channels. Default: [32, 64, 160, 256].
        depths (list[int], optional): Depths of each MSCAN stage.
        atten_kernel_sizes (list[int], optional): The height or width of each strip convolution kernel in attention module.
            Default: [7, 11, 21].
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float, optional): Dropout rate. Default: 0.0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.1.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 in_channels=1,
                 embed_dims=[16, 32, 80, 128], # [32, 64, 160, 256]
                 depths=[1, 1, 1, 1], # [3, 3, 5, 2]
                 mlp_ratios=[1, 1, 1, 1], # [8, 8, 4, 4]
                 atten_kernel_sizes=[1, 1, 1],# [7, 11, 21]
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 pretrained=None):
        super().__init__()
        if not _check_length(embed_dims, mlp_ratios, depths):
            raise ValueError(
                "The length of aurgments 'embed_dims', 'mlp_ratios' and 'drop_path_rate' must be same."
            )

        self.depths = depths
        self.num_stages = len(embed_dims)
        self.feat_channels = embed_dims

        drop_path_rates = [
            x for x in np.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(self.num_stages):
            if i == 0:
                patch_embed = StemConv(in_channels, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=3,
                    stride=2,
                    in_chans=embed_dims[i - 1],
                    embed_dim=embed_dims[i])

            block = [
                Block(
                    dim=embed_dims[i],
                    atten_kernel_sizes=atten_kernel_sizes,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=drop_path_rates[cur + j])
                for j in range(depths[i])
            ]
            norm = nn.LayerNorm(embed_dims[i]).to(device)
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        #self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            utils.load_pretrained_model(self, pretrained)
        else:
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.LayerNorm):
                    zeros_(sublayer.bias)
                    ones_(sublayer.weight)
                elif isinstance(sublayer, nn.Conv2D):
                    fan_out = (sublayer._kernel_size[0] *
                               sublayer._kernel_size[1] *
                               sublayer._out_channels)
                    fan_out //= sublayer._groups
                    initializer = Normal(mean=0, std=math.sqrt(2.0 / fan_out))
                    initializer(sublayer.weight)
                    zeros_(sublayer.bias)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x)

        return outs


def MSCAN_T(**kwargs):
    return MSCAN(**kwargs)


def MSCAN_S(**kwargs):
    return MSCAN(embed_dims=[16, 32, 64, 128], depths=[1, 1, 1, 1], **kwargs)
    return MSCAN(embed_dims=[64, 128, 320, 512], depths=[2, 2, 4, 2], **kwargs)


def MSCAN_B(**kwargs):
    return MSCAN(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 3, 6, 3],
        drop_path_rate=0.1,
        **kwargs)


def MSCAN_L(**kwargs):
    return MSCAN(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 5, 27, 3],
        drop_path_rate=0.3,
        **kwargs)

































































class SegNeXt(nn.Module):
    """
    The SegNeXt implementation based on PaddlePaddle.
    The original article refers to
    Guo, Meng-Hao, et al. "SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation"
    (https://arxiv.org/pdf/2209.08575.pdf)
    Args:
        backbone (nn.Layer): The backbone must be an instance of MSCAN.
        decoder_cfg (dict): The arguments of decoder.
        num_classes (int): The unique number of target classes.
        backbone_indices (list(int), optional): The values indicate the indices of backbone output
           used as the input of the SegNeXt head. Default: [1, 2, 3].
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 backbone,
                 decoder_cfg,
                 num_classes,
                 backbone_indices=[2, 2, 2], # [1, 2, 3]
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.name = "SegNext_"
        in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]
        self.decode_head = LightHamHead(
            in_channels=in_channels, num_classes=num_classes)

        self.align_corners = self.decode_head.align_corners
        self.pretrained = pretrained
        #self.init_weights()

    def init_weights(self):
        if self.pretrained:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        input_size = x.shape[2:]
        feats = self.backbone(x)
        out = self.decode_head(feats)

        return [
            F.interpolate(
                out,
                input_size,
                mode="bilinear",
                align_corners=self.align_corners)
        ][0]


class Hamburger(nn.Module):
    def __init__(self, ham_channels=512, num_groups=1, ham_kwargs=None):
        super().__init__()
        self.ham_in = nn.Conv2d(ham_channels, ham_channels, kernel_size=1).to(device)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvGNAct(
            ham_channels,
            ham_channels,
            kernel_size=1,
            num_groups=num_groups,
            bias_attr=False)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


class LightHamHead(nn.Module):
    """The head implementation of HamNet based on PaddlePaddle.
    The original article refers to Zhengyang Geng, et al. "Is Attention Better Than Matrix Decomposition?"
    (https://arxiv.org/abs/2109.04553.pdf)
    Args:
        in_channels (list[int]): The feature channels from backbone.
        num_classes (int): The unique number of target classes.
        channels (int, optional): The intermediate channel of LightHamHead. Default: 256.
        dropout_rate (float, optional): The rate of dropout. Default: 0.1.
        align_corners (bool, optional): Whether use align_corners when interpolating. Default: False.
        ham_channels (int, optional): Input channel of Hamburger. Default: 512.
        num_groups (int, optional): The num_groups of convolutions in LightHamHead. Default: 32.
        ham_kwargs (dict, optional): Keyword arguments of Hamburger module.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 channels=64, #256
                 dropout_rate=0.1,
                 align_corners=False,
                 ham_channels=64, # 512
                 num_groups=1, # 32
                 ham_kwargs=None):
        super().__init__()

        if len(in_channels) != 3:
            raise ValueError(
                "The length of `in_channels` must be 3, but got {}".format(
                    len(in_channels)))

        self.align_corners = align_corners

        self.squeeze = ConvGNAct(
            sum(in_channels),
            ham_channels,
            kernel_size=1,
            num_groups=num_groups,
            act_type="relu",
            bias_attr=False)

        self.hamburger = Hamburger(ham_channels, num_groups, ham_kwargs)

        self.align = ConvGNAct(
            ham_channels,
            channels,
            kernel_size=1,
            num_groups=num_groups,
            act_type="relu",
            bias_attr=False)

        self.dropout = (nn.Dropout2d(dropout_rate)
                        if dropout_rate > 0.0 else nn.Identity())
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1).to(device)

    def forward(self, a):
        inputs = a[1:]
        target_shape = inputs[0].shape[2:]
        inputs = [
            F.interpolate(
                level,
                size=target_shape,
                mode="bilinear",
                align_corners=self.align_corners) for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)
        del inputs
        x = self.hamburger(x)

        output = self.align(x)
        output = self.dropout(output)
        output = self.conv_seg(output)
        return output