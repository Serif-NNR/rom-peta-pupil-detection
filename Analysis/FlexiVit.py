# https://github.com/bwconrad/flexivit











































from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functorch import vmap
from torch import Tensor





def pi_resize_patch_embed(
    patch_embed: Tensor,
    new_patch_size: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """Resample patch embedding weights to a target resolution via pseudo-inverse
    resizing.
    Based on:
        https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py
        https://arxiv.org/abs/2212.08013
    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [d, c h', w']
    """
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4D tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (height, width)"

    old_patch_size = tuple(patch_embed.shape[2:])

    # Return original kernel if no resize is necessary
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x: Tensor, shape: Tuple[int, int]):
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=interpolation,
            antialias=antialias,
        )
        return x_resized[0, 0, ...]

    def calculate_pinv(old_shape: Tuple[int, int], new_shape: Tuple[int, int]):
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    # Calculate pseudo-inverse of resize matrix
    resize_matrix_pinv = calculate_pinv(old_patch_size, new_patch_size)
    resize_matrix_pinv = resize_matrix_pinv.to(patch_embed.device)

    def resample_patch_embed(patch_embed: Tensor):
        h, w = new_patch_size
        resampled_kernel = resize_matrix_pinv @ patch_embed.reshape(-1)
        return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

    v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

    return v_resample_patch_embed(patch_embed)


def interpolate_resize_patch_embed(
    patch_embed: Tensor,
    new_patch_size: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """Resample patch embedding weights to a target resolution via interpolation
    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [d, c h', w']
    """
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4D tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (height, width)"

    patch_embed = F.interpolate(
        patch_embed, new_patch_size, mode=interpolation, antialias=antialias
    )

    return patch_embed


class FlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 240,
        patch_size: Union[int, Tuple[int, int]] = 32,
        grid_size: Union[int, Tuple[int, int]] = 7,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
        bias: bool = True,
        patch_size_seq: Sequence[int] = (8, 10, 12, 15, 16, 20, 24, 30, 40, 48),
        patch_size_probs: Optional[Sequence[float]] = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """2D image to patch embedding w/ flexible patch sizes
        Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
        Args:
            img_size: Input image size
            patch_size: Base patch size. i.e the size of the parameter buffer
            grid_size: Size of pos_embed buffer
            in_chans: Number of input image channels
            embed_dim: Network embedding dimension size
            norm_layer: Optional normalization layer
            flatten: Whether to flatten the spatial dimensions of the output
            bias: Whether to use bias in convolution
            patch_size_seq: List of patch sizes to randomly sample from
            patch_size_probs: Optional list of probabilities to sample corresponding
                patch_size_seq elements. If None, then uniform distribution is used
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """
        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = to_2tuple(grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias

        self.patch_size_seq = patch_size_seq

        if self.patch_size_seq:
            if not patch_size_probs:
                n = len(self.patch_size_seq)
                self.patch_size_probs = [1.0 / n] * n
            else:
                self.patch_size_probs = [
                    p / sum(patch_size_probs) for p in patch_size_probs
                ]
        else:
            self.patch_size_probs = []

        # Pre-calculate pinvs
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        """Pre-calculate all pinv matrices"""
        pinvs = {}
        for ps in self.patch_size_seq:
            ps = to_2tuple(ps)
            pinvs[ps] = self._calculate_pinv(self.patch_size, ps)
        return pinvs

    def _resize(self, x: Tensor, shape: Tuple[int, int]) -> Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(
        self, old_shape: Tuple[int, int], new_shape: Tuple[int, int]
    ) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(self, patch_embed: Tensor, new_patch_size: Tuple[int, int]):
        """Resize patch_embed to target resolution via pseudo-inverse resizing"""
        # Return original kernel if no resize is necessary
        if self.patch_size == new_patch_size:
            return patch_embed

        # Calculate pseudo-inverse of resize matrix
        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(
                self.patch_size, new_patch_size
            )
        pinv = self.pinvs[new_patch_size]
        pinv = pinv.to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor):
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

        return v_resample_patch_embed(patch_embed)

    def forward(
        self,
        x: Tensor,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        return_patch_size: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[int, int]]]:

        if not patch_size and not self.training:
            # During evaluation use base patch size if not specified
            patch_size = self.patch_size
        elif not patch_size:
            # During training choose uniformly at random if not specified
            assert (
                self.patch_size_seq
            ), "No patch size specified during forward and no patch_size_seq given to FlexiPatchEmbed"
            patch_size = np.random.choice(self.patch_size_seq, p=self.patch_size_probs)

        patch_size = to_2tuple(patch_size)

        # Resize conv weights
        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)

        # Apply conv with resized weights
        x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)

        if return_patch_size:
            return x, patch_size

        return x









































































import collections.abc
import math
from itertools import repeat
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def to_2tuple(x: Any) -> Tuple:
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


def resize_abs_pos_embed(
    pos_embed: torch.Tensor,
    new_size: Tuple[int, int],
    old_size: Optional[Union[int, Tuple[int, int]]] = None,
    num_prefix_tokens: int = 1,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> torch.Tensor:
    """Resize absolute position embeddings to a target resolution via interpolation
    Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py
    Args:
        pos_embed: Position embeddings tensor of size [b, n, d]
        new_size: Target [height, width] of embedding
        old_size: Original [height, width] of embedding
        num_prefix_tokens: Number of non-spatial prefix tokens (eg. cls)
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [b, n', d]
    """

    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]

    if not old_size:
        old_size = int(math.sqrt(pos_embed.shape[1] - num_prefix_tokens))  # type:ignore
    old_size = to_2tuple(old_size)

    # Return if no resize necessary
    if new_size == old_size:
        return pos_embed

    if num_prefix_tokens:
        posemb_prefix, pos_embed = (
            pos_embed[:, :num_prefix_tokens],
            pos_embed[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, pos_embed = None, pos_embed

    # Interpolate position embedding
    pos_embed = pos_embed.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(
        pos_embed, size=new_size, mode=interpolation, antialias=antialias
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    # Add back extra prefix tokens
    if posemb_prefix is not None:
        pos_embed = torch.cat([posemb_prefix, pos_embed], dim=1)

    return pos_embed


















































from functools import partial
from typing import Optional, Sequence, Tuple, Union

import torch
from timm.models.vision_transformer import Block, VisionTransformer
from torch import Tensor, nn


class FlexiVisionTransformer(VisionTransformer):
    def __init__(
        self,
        img_size: int = 240,
        base_patch_size: Union[int, Tuple[int, int]] = 16, # 32,
        in_chans: int = 1, # 3,
        num_classes: int = 1, # 1000,
        global_pool: str = "token",
        embed_dim: int = 768, # 768,
        depth: int = 12, # 12,
        num_heads: int = 12, # 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = True,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        weight_init: str = "",
        embed_layer: nn.Module = FlexiPatchEmbed,  # type:ignore
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
        block_fn: nn.Module = Block,  # type:ignore
        patch_size_seq: Sequence[int] = (8, 10, 12, 15, 16, 20, 24, 30, 40, 48),
        base_pos_embed_size: Union[int, Tuple[int, int]] = 7,
        patch_size_probs: Optional[Sequence[float]] = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """Vision transformer w/ flexible patch sizes
        From: https://arxiv.org/abs/2212.08013
        Args:
            img_size: input image size
            patch_size: patch size
            in_chans: number of input channels
            num_classes: number of classes for classification head
            global_pool: type of global pooling for final sequence (default: 'token')
            embed_dim: embedding dimension
            depth: depth of transformer
            num_heads: number of attention heads
            mlp_ratio: ratio of mlp hidden dim to embedding dim
            qkv_bias: enable bias for qkv if True
            init_values: layer-scale init values
            class_token: use class token
            fc_norm: pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate: dropout rate
            attn_drop_rate: attention dropout rate
            drop_path_rate: stochastic depth rate
            weight_init: weight init scheme
            embed_layer: patch embedding layer
            norm_layer: normalization layer
            act_layer: MLP activation layer
            patch_size_seq: List of patch sizes to randomly sample from
            base_pos_embed_size: Base position embedding size. i.e. the size of the parameter buffer
            patch_size_probs: Optional list of probabilities of sample corresponding
                patch_size_seq element. If None, then uniform distribution is used
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """

        assert embed_layer == FlexiPatchEmbed, "embed_layer should be a FlexiPatchEmbed"

        self.name = "FlexiVit_EXPECTED_"

        # Pre-initialize the flexi specific patch embed arguments
        embed_layer_fn = partial(
            FlexiPatchEmbed,
            patch_size_seq=patch_size_seq,
            patch_size_probs=patch_size_probs,
            grid_size=base_pos_embed_size,
            interpolation=interpolation,
            antialias=antialias,
        )

        # Position embedding resizing function
        self.resize_pos_embed = partial(
            resize_abs_pos_embed,
            old_size=base_pos_embed_size,
            interpolation=interpolation,
            antialias=antialias,
            num_prefix_tokens=1 if class_token and not no_embed_class else 0,
        )

        self.img_size = to_2tuple(img_size)

        super().__init__(
            img_size,
            base_patch_size,  # type:ignore
            in_chans,
            num_classes,
            global_pool,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            init_values,
            class_token,
            no_embed_class,
            pre_norm,
            fc_norm,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            weight_init,
            embed_layer_fn,  # type:ignore
            norm_layer,
            act_layer,
            block_fn,  # type:ignore
        )

    def _pos_embed(self, x: Tensor, patch_size: Tuple[int, int], x_size) -> Tensor:
        # Resize position embedding based on current patch size
        new_size = (
            int(x_size[2] // patch_size[0]),
            int(x_size[3] // patch_size[1]),
        )
        pos_embed = self.resize_pos_embed(self.pos_embed, new_size)

        if self.no_embed_class:
            # Position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # Position embedding has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def forward_features(
        self, x: Tensor, patch_size: Optional[Union[int, Tuple[int, int]]] = None
    ) -> Tensor:
        x_size = x.size()
        x, ps = self.patch_embed(x, patch_size, return_patch_size=True)
        x = self._pos_embed(x, ps, x_size)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(
        self, x: Tensor, patch_size: Optional[Union[int, Tuple[int, int]]] = None
    ) -> Tensor:
        x = self.forward_features(x, patch_size)
        x = self.forward_head(x)
        return x


def flexivit_tiny(**kwargs) -> FlexiVisionTransformer:
    return FlexiVisionTransformer(embed_dim=192, depth=12, num_heads=3, **kwargs)


def flexivit_small(**kwargs) -> FlexiVisionTransformer:
    return FlexiVisionTransformer(embed_dim=384, depth=12, num_heads=6, **kwargs)


def flexivit_base(**kwargs) -> FlexiVisionTransformer:
    return FlexiVisionTransformer(embed_dim=768, depth=12, num_heads=12, **kwargs)


def flexivit_large(**kwargs) -> FlexiVisionTransformer:
    return FlexiVisionTransformer(embed_dim=1024, depth=24, num_heads=16, **kwargs)


def flexivit_huge(**kwargs) -> FlexiVisionTransformer:
    return FlexiVisionTransformer(embed_dim=1280, depth=32, num_heads=16, **kwargs)














