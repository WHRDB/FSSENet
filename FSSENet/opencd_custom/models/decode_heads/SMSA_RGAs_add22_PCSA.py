import typing as t
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from mmengine.model import BaseModule


class SMSA_RGAs_add22_PCSA(BaseModule):

    def __init__(
        self,
        dim: int,
        head_num: int,
        window_size: int = 7,
        group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
        qkv_bias: bool = False,
        fuse_bn: bool = False,
        down_sample_mode: str = "avg_pool",
        attn_drop_ratio: float = 0.0,
        gate_layer: str = "sigmoid",
        in_channel=256,
        in_spatial=256,
        cha_ratio=8,
        spa_ratio=8,
        down_ratio=8,
    ):
        super(SMSA_RGAs_add22_PCSA, self).__init__()
        if dim <= 0 or dim % 4 or head_num <= 0 or dim % head_num:
            raise ValueError("dim must be divisible by 4 and head_num.")
        if in_channel != dim:
            raise ValueError("in_channel must equal dim.")
        if len(group_kernel_sizes) != 4 or any(
            k <= 0 or k % 2 == 0 for k in group_kernel_sizes
        ):
            raise ValueError(
                "group_kernel_sizes must contain four positive odd values."
            )
        if window_size != -1 and window_size <= 0:
            raise ValueError("window_size must be positive or -1 for global pooling.")
        if down_sample_mode not in ("avg_pool", "max_pool"):
            raise ValueError("down_sample_mode must be avg_pool or max_pool.")
        if gate_layer not in ("sigmoid", "softmax"):
            raise ValueError("gate_layer must be sigmoid or softmax.")
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** (-0.5)
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode
        self.group_chans = group_chans = self.dim // 4
        self.local_dwc = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[0],
            padding=group_kernel_sizes[0] // 2,
            groups=group_chans,
        )
        self.global_dwc_s = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[1],
            padding=group_kernel_sizes[1] // 2,
            groups=group_chans,
        )
        self.global_dwc_m = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[2],
            padding=group_kernel_sizes[2] // 2,
            groups=group_chans,
        )
        self.global_dwc_l = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[3],
            padding=group_kernel_sizes[3] // 2,
            groups=group_chans,
        )
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == "softmax" else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim
        )
        self.k = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim
        )
        self.v = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim
        )
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == "softmax" else nn.Sigmoid()
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        elif down_sample_mode == "avg_pool":
            self.down_func = nn.AvgPool2d(
                kernel_size=(window_size, window_size), stride=window_size
            )
        elif down_sample_mode == "max_pool":
            self.down_func = nn.MaxPool2d(
                kernel_size=(window_size, window_size), stride=window_size
            )
        self.in_channel = in_channel
        self.in_spatial = in_spatial
        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio
        self.gx_spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channel,
                out_channels=self.inter_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(),
        )
        self.gg_spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_spatial * 2,
                out_channels=self.inter_spatial,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU(),
        )
        num_channel_s = 1 + self.inter_spatial
        self.W_spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channel_s,
                out_channels=num_channel_s // down_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_channel_s // down_ratio),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_channel_s // down_ratio,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(1),
        )
        self.theta_spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channel,
                out_channels=self.inter_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(),
        )
        self.phi_spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channel,
                out_channels=self.inter_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The dim of x is (B, C, H, W)
        """
        b, c, h_, w_ = x.size()
        if c != self.dim or h_ * w_ != self.in_spatial:
            raise ValueError(
                "FSSENet attention expects 256 spatial positions. Use 512 x 512 crops with the supplied MiT-B0 configuration."
            )
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)
        x_h_attn = self.sa_gate(
            self.norm_h(
                torch.cat(
                    (
                        self.local_dwc(l_x_h),
                        self.global_dwc_s(g_x_h_s),
                        self.global_dwc_m(g_x_h_m),
                        self.global_dwc_l(g_x_h_l),
                    ),
                    dim=1,
                )
            )
        )
        x_h_attn = x_h_attn.view(b, c, h_, 1)
        x_w_attn = self.sa_gate(
            self.norm_w(
                torch.cat(
                    (
                        self.local_dwc(l_x_w),
                        self.global_dwc_s(g_x_w_s),
                        self.global_dwc_m(g_x_w_m),
                        self.global_dwc_l(g_x_w_l),
                    ),
                    dim=1,
                )
            )
        )
        x_w_attn = x_w_attn.view(b, c, 1, w_)
        x = x * x_h_attn * x_w_attn
        x1 = x
        b, c, h, w = x.size()
        theta_xs = self.theta_spatial(x)
        phi_xs = self.phi_spatial(x)
        theta_xs = theta_xs.view(b, self.inter_channel, -1)
        theta_xs = theta_xs.permute(0, 2, 1)
        phi_xs = phi_xs.view(b, self.inter_channel, -1)
        Gs = torch.matmul(theta_xs, phi_xs)
        Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)
        Gs_out = Gs.view(b, h * w, h, w)
        Gs_joint = torch.cat((Gs_in, Gs_out), 1)
        Gs_joint = self.gg_spatial(Gs_joint)
        g_xs = self.gx_spatial(x)
        g_xs = torch.mean(g_xs, dim=1, keepdim=True)
        ys = torch.cat((g_xs, Gs_joint), 1)
        W_ys = self.W_spatial(ys)
        x = F.sigmoid(W_ys.expand_as(x)) * x
        x = x1 + x
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        q = rearrange(
            q,
            "b (head_num head_dim) h w -> b head_num head_dim (h w)",
            head_num=int(self.head_num),
            head_dim=int(self.head_dim),
        )
        k = rearrange(
            k,
            "b (head_num head_dim) h w -> b head_num head_dim (h w)",
            head_num=int(self.head_num),
            head_dim=int(self.head_dim),
        )
        v = rearrange(
            v,
            "b (head_num head_dim) h w -> b head_num head_dim (h w)",
            head_num=int(self.head_num),
            head_dim=int(self.head_dim),
        )
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v
        attn = rearrange(
            attn,
            "b head_num head_dim (h w) -> b (head_num head_dim) h w",
            h=int(h_),
            w=int(w_),
        )
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x
