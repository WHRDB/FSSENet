import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, ConvModule, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, Sequential
from mmseg.models.utils import resize
from opencd.registry import MODELS
from .CA import CA
from .PAG import PagFM
from .SMSA_RGAs_add22_PCSA import SMSA_RGAs_add22_PCSA


class BridgeLayer(BaseModule):
    """Coordinate attention and pixel-wise fusion of foundation and side features."""

    def __init__(self):
        super().__init__()
        self.CA_layer = nn.ModuleList()
        self.CA_layer.append(CA(inp=256, oup=256, groups=4))
        self.PAG_layer = nn.ModuleList()
        self.PAG_layer.append(
            PagFM(in_channels=256, mid_channels=64, with_channel=True)
        )

    def forward(self, x, x_kv):
        hwshape = x.shape[-2:]
        x_kv = self.CA_layer[0](x_kv)
        x = self.PAG_layer[0](x, x_kv)
        x = x + resize(x_kv, size=hwshape, mode="bilinear", align_corners=False)
        return x


class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer.         Here MixFFN is uesd as projection head of Changer.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        feedforward_channels,
        act_cfg=dict(type="GELU"),
        ffn_drop=0.0,
        dropout_layer=None,
        init_cfg=None,
    ):
        super(MixFFN, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)
        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels,
        )
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        )

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


@MODELS.register_module()
class BAN_MLPDecoder(BaseModule):

    def __init__(
        self,
        in_channels,
        channels,
        num_classes,
        norm_cfg=None,
        dropout_ratio=0.1,
        act_cfg=dict(type="ReLU"),
        align_corners=False,
        interpolate_mode="bilinear",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        self.out_channels = num_classes
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )
        self.SMSA_RGAs_add22_PCSA_layer1 = nn.ModuleList()
        self.SMSA_RGAs_add22_PCSA_layer1.append(
            SMSA_RGAs_add22_PCSA(
                dim=256,
                head_num=1,
                window_size=7,
                group_kernel_sizes=[3, 5, 7, 9],
                qkv_bias=False,
                fuse_bn=False,
                down_sample_mode="avg_pool",
                attn_drop_ratio=0.0,
                gate_layer="sigmoid",
                in_channel=256,
                in_spatial=256,
                cha_ratio=8,
                spa_ratio=16,
                down_ratio=8,
            )
        )
        self.discriminator = MixFFN(
            embed_dims=self.channels * 2,
            feedforward_channels=self.channels * 2,
            ffn_drop=0.0,
            dropout_layer=dict(type="DropPath", drop_prob=0.0),
            act_cfg=dict(type="GELU"),
        )
        self.conv_seg = nn.Conv2d(self.channels * 2, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def base_forward(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners,
                )
            )
        out = self.fusion_conv(torch.cat(outs, dim=1))
        return out

    def forward(self, inputs1, inputs2):
        inputs1[3] = self.SMSA_RGAs_add22_PCSA_layer1[0](inputs1[3])
        inputs2[3] = self.SMSA_RGAs_add22_PCSA_layer1[0](inputs2[3])
        out1 = self.base_forward(inputs1)
        out2 = self.base_forward(inputs2)
        out = torch.cat([out1, out2], dim=1)
        out = self.discriminator(out)
        out = self.cls_seg(out)
        return out
