import timm
import torch
import torch.nn as nn
import timm
import torch.distributed as dist
import torch.nn.functional as F
import sys
from fvcore.nn.distributed import differentiable_all_reduce

import os
sys.path.append('.')

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .dwt_extractor import DwtFrequencyExtractor
from .dct_extractor import DctFrequencyExtractor
import math
from functools import partial
from IMDLBenCo.registry import MODELS

class ConvNeXt(timm.models.convnext.ConvNeXt):
    def __init__(self,conv_pretrain=False):
        super(ConvNeXt, self).__init__(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
        if conv_pretrain:
            print("Load Convnext pretrain.")
            model = timm.create_model('convnext_tiny', pretrained=True)
            self.load_state_dict(model.state_dict())
        original_first_layer = self.stem[0]
        new_first_layer = nn.Conv2d(6, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
        new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
        new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
        self.stem[0] = new_first_layer
        self.stages = self.stages[:-1]
        del self.head
    def forward_features(self, x):
        x = self.stem(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        x = self.norm_pre(x)
        return x , [out[0],out[2]]
    def forward(self, image, mask=None, *args, **kwargs):
        feature,out = self.forward_features(image)

        return feature,out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.float()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x) 
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) #(B, C, H*W) -> (B, H*W, C)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self,seg_pretrain_path=None, img_size=512, patch_size=4, in_chans=3,embed_dims=[64, 128, 320, 512],num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])

        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule 
        cur = 0
        # stage1 Transformer
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        if seg_pretrain_path is not None:
            self.load_state_dict(torch.load(seg_pretrain_path),
                              strict=False)
        original_first_layer = self.patch_embed1.proj
        new_first_layer = nn.Conv2d(6, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
        new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
    
        new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
        self.patch_embed1.proj = new_first_layer


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def stage1(self, B, x):
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W) 
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # reshape to (B, C, H, W)
        return x

    def stage2(self, B, x):
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x
    
    def stage3(self, B, x):
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def stage4(self, B, x):
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x) 
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
 
        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return x,outs[1:]

    def forward(self, x):
        x,outs = self.forward_features(x)
        return x,outs 


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class UpsampleConcatConv(nn.Module):
    def __init__(self):
        super(UpsampleConcatConv, self).__init__()
        self.upsamples2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.upsamplec3 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        )
        self.upsamples3 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )


        self.upsamples4 = nn.Sequential(
            nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )


    def forward(self, inputs):
        c1,c3,s2,s3,s4 = inputs

        c3 = self.upsamplec3(c3)
        s2 = self.upsamples2(s2)
        s3 = self.upsamples3(s3)
        s4 = self.upsamples4(s4)
        
        x = torch.cat([c1,c3,s2,s3,s4], dim=1)
        features = [c1,c3,s2,s3,s4 ]
        # shortcut = x
        # x = x.permute(0, 2, 3, 1)
        # x = self.fc2(self.act(self.fc1(x)))
        # x = x.permute(0, 3, 1, 2)
        # x = x + shortcut
        return x, features

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    
class ScoreNetwork(nn.Module):
    def __init__(self):
        super(ScoreNetwork, self).__init__()
        self.conv1 = nn.Conv2d(9, 192, kernel_size=7, stride=2, padding=3)
        self.invert = nn.Sequential(LayerNorm2d(192),
                                    nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(192, 768, kernel_size=1),
                                    nn.Conv2d(768, 192, kernel_size=1),
                                    nn.GELU())
        self.conv2 = nn.Conv2d(192, 8,  kernel_size=7, stride=2, padding=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        short_cut = x
        x = self.invert(x)
        x = short_cut + x
        x = self.conv2(x)
        x = x.float()
        x = self.softmax(x)
        return x



class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=3):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num)) # initialize as 0?

    def forward(self, *losses):
        # losses: [loss1, loss2]
        total_loss = 0
        for loss, log_var in zip(losses, self.params):
            precision = torch.exp(-log_var)
            total_loss += precision * loss + 0.5 * log_var
        return total_loss

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

import warnings
class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            with warnings.catch_warnings(record=True):
                if x.numel() == 0 and self.training:
                    # https://github.com/pytorch/pytorch/issues/12013
                    assert not isinstance(
                        self.norm, torch.nn.SyncBatchNorm
                    ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SimpleFeaturePyramid(nn.Module):
    """
    An sequetial implementation of Simple-FPN in 'vitdet' paper.
    """
    def __init__(self,
        in_channels,
        out_channels,
        scale_factors,
        input_stride = 16,
        top_block=None,
        norm=None
    ) -> None:
        super().__init__()
        
        dim = in_channels
        self.dim = dim
        self.scale_factors = scale_factors
        # self.stages = []
        self.stages = nn.ModuleList()
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    torch.nn.Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    LayerNorm(out_channels),
                    torch.nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    LayerNorm(out_channels)

                ]
            )
            layers = nn.Sequential(*layers)
            self.stages.append(layers)

    def forward(self, features):
        results = []
        for stage in self.stages:
            # print(features.device)
            # exit(0)
            results.append(stage(features))
        results.append(F.max_pool2d(results[0], kernel_size=1, stride=2, padding=0))
        return results


class LastLevelMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]



class PredictHead(nn.Module):
    def __init__(self, 
                feature_channels : list,
                embed_dim = 256,
                predict_channels : int = 1,
                norm : str = "BN"
                ) -> None:
        """
        We tested three different types of normalization in the decoder head, and they may yield different results due to dataset configurations and other factors.
        Some intuitive conclusions are as follows:
            - "LN" -> Layer norm : The fastest convergence, but poor generalization performance.
            - "BN" Batch norm : When include authentic images during training, set batchsize = 2 may have poor performance. But if you can train with larger batchsize (e.g. A40 with 48GB memory can train with batchsize = 4) It may performs better.
            - "IN" Instance norm : A form that can definitely converge, equivalent to a batchnorm with batchsize=1. When abnormal behavior is observed with BatchNorm, one can consider trying Instance Normalization. It's important to note that in this case, the settings should include setting track_running_stats and affine to True, rather than the default settings in PyTorch.
        """
        
        super().__init__()
        c1_in_channel, c2_in_channel, c3_in_channel, c4_in_channel, c5_in_channel = feature_channels
        assert len(feature_channels) == 5 , "feature_channels must be a list of 5 elements"
        # self.linear_c5 = MLP(input_dim = c5_in_channel, output_dim = embed_dim)
        # self.linear_c4 = MLP(input_dim = c4_in_channel, output_dim = embed_dim)
        # self.linear_c3 = MLP(input_dim = c3_in_channel, output_dim = embed_dim)
        # self.linear_c2 = MLP(input_dim = c2_in_channel, output_dim = embed_dim)
        # self.linear_c1 = MLP(input_dim = c1_in_channel, output_dim = embed_dim)

        self.linear_fuse = nn.Conv2d(
            in_channels= embed_dim * 5,
            out_channels= embed_dim,
            kernel_size= 1
        )
        
        assert norm in ["LN", "BN", "IN"], "Argument error when initialize the predict head : Norm argument should be one of the 'LN', 'BN' , 'IN', which represent Layer_norm, Batch_norm and Instance_norm"
        
        if norm == "LN":
            self.norm = LayerNorm(embed_dim)
        elif norm == "BN" :
            self.norm = nn.BatchNorm2d(embed_dim)
        else:
            self.norm = nn.InstanceNorm2d(embed_dim, track_running_stats=True, affine=True)

        self.dropout = nn.Dropout()
        
        self.linear_predict = nn.Conv2d(embed_dim, predict_channels, kernel_size= 1)
        
    def forward(self, x):
        c1, c2, c3, c4, c5 = x    # 1/4 1/8 1/16 1/32 1/64
        
        n, _ , h, w = c1.shape # Target size of all the features
        
        # _c1 = self.linear_c1(c1).reshape(shape=(n, -1, c1.shape[2], c1.shape[3]))
        
        _c1 =  F.interpolate(c1, size=(h, w), mode='bilinear', align_corners=False)
        
        # _c2 = self.linear_c2(c2).reshape(shape=(n, -1, c2.shape[2], c2.shape[3]))
        
        _c2 = F.interpolate(c2, size=(h, w), mode='bilinear', align_corners=False)
        
        # _c3 = self.linear_c3(c3).reshape(shape=(n, -1, c3.shape[2], c3.shape[3]))
        
        _c3 = F.interpolate(c3, size=(h, w), mode='bilinear', align_corners=False)
        
        # _c4 = self.linear_c4(c4).reshape(shape=(n, -1, c4.shape[2], c4.shape[3]))
        
        _c4 = F.interpolate(c4, size=(h, w), mode='bilinear', align_corners=False) 
        
        # _c5 = self.linear_c5(c5).reshape(shape=(n, -1, c5.shape[2], c5.shape[3]))
        
        _c5 = F.interpolate(c5, size=(h, w), mode='bilinear', align_corners=False)
        
        _c = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4, _c5], dim=1))
        
        _c = self.norm(_c)
        
        x = self.dropout(_c)
        
        x = self.linear_predict(x)
        
        return x


@MODELS.register_module()
class AniXplore(nn.Module):
    def __init__(self, seg_pretrain_path, conv_pretrain=False, freeze_segmodel=False, image_size=512):
        super(AniXplore, self).__init__()
        self.convnext = ConvNeXt(conv_pretrain)
        self.segformer = MixVisionTransformer(seg_pretrain_path)
        self.dct = DctFrequencyExtractor()
        self.high_dwt = DwtFrequencyExtractor()
        self.resize = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=True)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(96 + 64, 96, kernel_size=1),  #  1-128x128 
            nn.Conv2d(192 + 128, 192, kernel_size=1),  #  2-64x64 
            nn.Conv2d(384 + 320, 384, kernel_size=1),  # 3-32x32 
        ])


        self.predict_head = PredictHead(
            feature_channels=[256 for i in range(5)],
            embed_dim=256,
            norm="BN"  # important! may influence the results
        )
        self.featurePyramid_net = SimpleFeaturePyramid(
            in_channels=384,
            out_channels=256,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            top_block=None,
            norm="LN",
            )
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(384, 1) # combine 5 maps, channel = 384
        )
        self.auto_weight = AutomaticWeightedLoss(num=2)

    def forward(self, image, mask, label, *args, **kwargs):
        high_dct_freq = self.dct.forward_high(image)
        high_dwt_freq = self.high_dwt.forward(image)
        high_freq = high_dct_freq * 0.5 + high_dwt_freq * 0.5
        # print(high_freq.shape)
        low_freq = self.dct.forward_low(image)
        # print(low_freq.shape)
        input_high = torch.concat([image,high_freq],dim=1)
        input_low = torch.concat([image,low_freq],dim=1)

        B = input_low.shape[0] # get batch size
        x = self.convnext.stem(input_high)
        x = self.convnext.stages[0](x)
        y = self.segformer.stage1(B, input_low)
        fused_feat = torch.cat([x, y], dim=1)
        fused_feat = self.fusion_layers[0](fused_feat)


        x = self.convnext.stages[1](fused_feat)
        y = self.segformer.stage2(B, y)
        fused_feat = torch.cat([x, y], dim=1)
        fused_feat = self.fusion_layers[1](fused_feat)

        x = self.convnext.stages[2](fused_feat)
        y = self.segformer.stage3(B, y)
        fused_feat = torch.cat([x, y], dim=1)
        # [b, 384, 32, 32]
        fused_feat = self.fusion_layers[2](fused_feat)

        multiscale_feat = self.featurePyramid_net(fused_feat)

        pred_mask = self.predict_head(multiscale_feat)
        pred_mask = self.resize(pred_mask)
        loss = self.loss_fn(pred_mask,mask)
        pred_mask = pred_mask.float()
        mask_pred = torch.sigmoid(pred_mask)


        raw_cls_logit = self.cls_head(fused_feat)
        label = label.float()
        cls_loss = F.binary_cross_entropy_with_logits(raw_cls_logit[:, -1, ...], label)
        #  get pred_label
        # print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(f'raw_cls_logit={raw_cls_logit}')
        # cls_logit = torch.softmax(raw_cls_logit, dim=1)
        # pred_label = cls_logit[:, -1, ...]
        # print(f'cls_logit={cls_logit}')
        # print(f'pred_label={pred_label}')
        pred_label_prob_2d = torch.sigmoid(raw_cls_logit)  # Shape: [batch_size, 1]
        pred_label_prob_1d = pred_label_prob_2d.squeeze(-1)    # Shape: [batch_size]
        pred_label_binary = (pred_label_prob_1d > 0.5).float()
        
        # print(f'pred_label_prob = {pred_label_prob}')
        # print(f'pred_label_binary = {pred_label_binary}')
        
        combined_loss = self.auto_weight(loss, cls_loss)

        output_dict = {
            # loss for backward
            "backward_loss": combined_loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": mask_pred,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": pred_label_binary,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                "predict_loss": loss,
                "cls_loss": cls_loss,
                "combined_loss": combined_loss,
            },

            "visual_image": {
                "pred_mask": mask_pred,
            }
            # -----------------------------------------
        }
        return output_dict