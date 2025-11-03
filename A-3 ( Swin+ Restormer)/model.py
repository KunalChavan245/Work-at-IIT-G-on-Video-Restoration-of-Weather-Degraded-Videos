import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class BPWFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(3))
        self.radius = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        out = []
        for c in range(3):
            t = (self.theta[c] + 1) * torch.pi / 2
            r = self.radius[c]
            p1 = torch.tensor([r * torch.cos(t), r * torch.sin(t)], device=x.device)
            p2 = torch.tensor([1 - p1[0], 1 - p1[1]], device=x.device)
            x_c = x[:, c:c+1]
            x_out = 3 * (1 - x_c)**2 * x_c * p1[1] + 3 * (1 - x_c) * x_c**2 * p2[1] + x_c**3
            out.append(x_out)
        return torch.cat(out, dim=1)

class KBLFilter(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.k1 = nn.Parameter(torch.rand(3, 1, kernel_size, kernel_size) * 2 - 1)
        self.k2 = nn.Parameter(torch.rand(3, 1, kernel_size, kernel_size) * 2 - 1)
        self.groups = 3
        self.pad = kernel_size // 2

    def forward(self, x):
        out = x + F.conv2d(x, self.k1, padding=self.pad, groups=self.groups) * x + \
                  F.conv2d(x, self.k2, padding=self.pad, groups=self.groups)
        return out

class FilterParameterEncoder(nn.Module):
    def __init__(self, out_dim=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)

class WeatherPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FilterParameterEncoder()
        self.kbl = KBLFilter()
        self.bpw = BPWFilter()

    def forward(self, x):
        params = self.encoder(x)
        self.bpw.theta.data = params[:, :3].mean(dim=0)
        self.bpw.radius.data = params[:, 3:].mean(dim=0)
        x = self.bpw(x)
        x = self.kbl(x)
        return x

class GradientBranch(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).repeat(3, 1, 1, 1))

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        return torch.cat([grad_x, grad_y], dim=1)


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r):
        super(SCConv, self).__init__()

        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=1, dilation=dilation,
                                groups=groups, bias=False),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, dilation=dilation,
                                groups=1, bias=False),
                    )

    def forward(self, x):
        identity = x
        out = torch.sigmoid(F.interpolate(self.k2(x), identity.size()[2:]))
        out = torch.mul(self.k3(x), out)
        out = self.k4(out)
        return out

class Temporal_Alignment(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 act_layer=nn.ReLU()):
        super(Temporal_Alignment, self).__init__()

        self.offset_channels_3x3 = 2 * 3 * 3
        self.offset_channels_5x5 = 2 * 5 * 5
        self.offset_channels_7x7 = 2 * 7 * 7
        total_offset_channels = self.offset_channels_3x3 + self.offset_channels_5x5 + self.offset_channels_7x7

        self.offset_conv1 = SCConv(in_channels*2, total_offset_channels, 1, padding, dilation, 1, 2)

        self.deform_3x3 = DeformConv2d(in_channels, out_channels, 3, padding=1, groups=8)
        self.deform_5x5 = DeformConv2d(in_channels, out_channels, 5, padding=2, groups=8)
        self.deform_7x7 = DeformConv2d(in_channels, out_channels, 7, padding=3, groups=8)

        self.act_layer = act_layer
        self.pointwise = torch.nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x, y):
        all_offsets = self.offset_conv1(torch.cat((x, y), dim=1))

        start = 0
        end = self.offset_channels_3x3
        offset_3x3 = all_offsets[:, start:end, :, :]

        start = end
        end += self.offset_channels_5x5
        offset_5x5 = all_offsets[:, start:end, :, :]

        start = end
        end += self.offset_channels_7x7
        offset_7x7 = all_offsets[:, start:end, :, :]

        feat1_3x3 = self.deform_3x3(x, offset_3x3)
        feat1_5x5 = self.deform_5x5(x, offset_5x5)
        feat1_7x7 = self.deform_7x7(x, offset_7x7)

        feat1_3x3 = self.act_layer(feat1_3x3)
        feat1_5x5 = self.act_layer(feat1_5x5)
        feat1_7x7 = self.act_layer(feat1_7x7)

        concat_features = torch.cat([feat1_3x3, feat1_5x5, feat1_7x7], dim=1)
        x = self.pointwise(concat_features)

        return x

class Multi_Receptive_Attentive_Prompts(nn.Module):
    def __init__(self, channel_dim=32):
        super(Multi_Receptive_Attentive_Prompts, self).__init__()
        self.conv1x1 = nn.Conv2d(channel_dim*2, channel_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(channel_dim*3, channel_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.local_prompt_extractor1 = nn.Conv2d(channel_dim, channel_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.local_prompt_extractor2 = nn.Conv2d(channel_dim, channel_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.Global_Prompt_Extractor = nn.Conv2d(channel_dim, channel_dim, kernel_size=1)


    def forward(self, x):
        x_local_1 = self.local_prompt_extractor1(x)
        x_local_2 = self.local_prompt_extractor2(x)
        x_global = self.Global_Prompt_Extractor(x)
        concat_feature = torch.cat([x_local_1, x_local_2, x_global], dim=1)
        prompt = self.conv3x3(concat_feature)
        return prompt

class DeformableWindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8, learnable_temp=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.learnable_temp = learnable_temp

        self.qkv = nn.Linear(dim * window_size * window_size, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.window_reconstruct = nn.Linear(dim, dim * window_size * window_size)

        self.offset_conv = nn.Conv2d(dim, 2 * window_size * window_size, kernel_size=3, padding=1, bias=True)
        self.unfold = nn.Unfold(kernel_size=window_size, stride=window_size)
        self.fold = nn.Fold(output_size=(1, 1), kernel_size=window_size, stride=window_size)

    def forward(self, x):
        B, C, H, W = x.shape
        self.fold.output_size = (H, W)

        offsets = self.offset_conv(x)

        windows = self.unfold(x).transpose(1, 2)

        qkv = self.qkv(windows)
        qkv = qkv.reshape(B, -1, self.num_heads, 3 * self.head_dim)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.learnable_temp is not None:
            attn = attn * self.learnable_temp

        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, -1, self.dim)

        out = self.proj(out)

        out_reconstructed_windows = self.window_reconstruct(out)
        out_folded_input = out_reconstructed_windows.transpose(1, 2)

        out = self.fold(out_folded_input)

        return out

class HybridAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local_attn = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        local = self.local_attn(x)
        channel = x * self.channel_attn(x)
        concat = torch.cat([local, channel], dim=1)
        return self.out_proj(concat)

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = DeformableWindowAttention(dim=channels, window_size=8, num_heads=num_heads)

    def forward(self, x):
        b, c, h, w = x.shape
        x_normed = self.norm(x.view(b, c, -1)).view(b, c, h, w)
        x = x + self.attn(x_normed)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super(TransformerBlock, self).__init__()
        self.attn = MDTA(channels, num_heads)

    def forward(self, x):
        return self.attn(x)

class MDTA_FOR_VIDEO(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.temporal_align = Temporal_Alignment(channels, channels, 3)
        self.norm = nn.GroupNorm(1, channels)
        self.swinir_core = SwinIRBlockForVideo(dim=channels, input_resolution=(128,128), # Placeholder, will be updated dynamically
                                          num_heads=num_heads, window_size=8)


    def forward(self, x, y):
        # x is current frame feature, y is recurrent frame feature
        aligned_y = self.temporal_align(y, x) # Align previous frame feature to current frame
        
        # Concatenate current frame feature with aligned previous frame feature
        # This merged feature then goes into the SwinIR-like core
        merged_feature = x + aligned_y # Simple addition for fusion, can be concat then conv

        b, c, h, w = merged_feature.shape
        merged_feature_normed = self.norm(merged_feature.view(b, c, -1)).view(b, c, h, w)

        # Update input_resolution for the SwinIR block
        self.swinir_core.update_input_resolution((h, w))

        # Pass the merged feature through the SwinIR-like core
        # SwinIRBlockForVideo expects B, H*W, C. Need to flatten and unflatten.
        output_feature = self.swinir_core(merged_feature_normed.flatten(2).transpose(1, 2))
        
        return output_feature.transpose(1, 2).view(b, c, h, w) # Convert back to B, C, H, W

class MDTA_FOR_VIDEO_Prompted(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.prompt_block = Multi_Receptive_Attentive_Prompts(channel_dim=channels)
        self.temporal_align = Temporal_Alignment(channels, channels, 3)
        # FIX: GroupNorm input channels to match actual concatenated feature size (channels * 3)
        self.norm = nn.GroupNorm(1, channels * 3) 
        # FIX: SwinIRBlockForVideo input dim to match actual concatenated feature size (channels * 3)
        self.swinir_core = SwinIRBlockForVideo(dim=channels * 3, input_resolution=(128,128), # Placeholder, will be updated dynamically
                                          num_heads=num_heads, window_size=8)
        # FIX: Fusion layer input channels to match actual concatenated feature size (channels * 3)
        self.fusion = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)

    def forward(self, x, y):
        b, c, h, w = x.shape
        prompt = self.prompt_block(x)
        x_prompted = torch.cat([x, prompt], dim=1) # Original feature (channels) + prompt (channels) = 2*channels

        aligned_y = self.temporal_align(y, x) # Aligned previous frame feature (channels)

        # Concatenate x_prompted (2*channels) and aligned_y (channels) = 3*channels
        merged_feature = torch.cat([x_prompted, aligned_y], dim=1) 

        # Normalize and pass through SwinIR core
        merged_feature_normed = self.norm(merged_feature.view(b, merged_feature.shape[1], -1)).view(b, merged_feature.shape[1], h, w)

        # Update input_resolution for the SwinIR block
        self.swinir_core.update_input_resolution((h, w))
        
        # Pass the merged feature through the SwinIR-like core
        output_feature = self.swinir_core(merged_feature_normed.flatten(2).transpose(1, 2))

        fused_output = self.fusion(output_feature.transpose(1, 2).view(b, -1, h, w)) # Final fusion

        return fused_output


class TransformerBlock_for_Video(nn.Module):
    def __init__(self, channels, num_heads):
        super(TransformerBlock_for_Video, self).__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.attn_and_swinir = MDTA_FOR_VIDEO(channels, num_heads)
        self.hybrid_attn = HybridAttention(channels)

    def forward(self, x, y):
        assert x.shape[1] == y.shape[1], f"Channel mismatch: x={x.shape}, y={y.shape}"
        
        output_from_swinir_and_temporal = self.attn_and_swinir(x, y)
        
        x = x + self.hybrid_attn(output_from_swinir_and_temporal)
        return x

class TransformerBlock_for_Video_Prompted(nn.Module):
    def __init__(self, channels, num_heads):
        super(TransformerBlock_for_Video_Prompted, self).__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.attn_and_swinir = MDTA_FOR_VIDEO_Prompted(channels, num_heads)
        self.hybrid_attn = HybridAttention(channels)

    def forward(self, x, y):
        assert x.shape[1] == y.shape[1], f"Channel mismatch: x={x.shape}, y={y.shape}"
        
        output_from_swinir_and_temporal = self.attn_and_swinir(x, y)
        
        x = x + self.hybrid_attn(output_from_swinir_and_temporal)
        return x


class DownSample(nn.Module):
    def __init__(self, channels, channels_out):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels_out, kernel_size=3, padding=1, bias=False, stride=2))

    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.gated_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.gated_conv2d(x)
        return x

class BottleneckTemporalFusion(nn.Module):
    def __init__(self, in_out_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_out_channels * 2, in_out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, prev):
        return self.fusion(torch.cat([x, prev], dim=1))

class DualPathFusion(nn.Module):
    def __init__(self, low_res_ch, high_res_ch, out_ch):
        super().__init__()
        # Input to conv is low_res_ch + high_res_ch
        self.conv = nn.Sequential(
            nn.Conv2d(low_res_ch + high_res_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, low_res_feat, high_res_feat):
        up_high = F.interpolate(high_res_feat, size=low_res_feat.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([low_res_feat, up_high], dim=1))


#####################################################################
#           SwinIR-like Components for Replacement                  #
#####################################################################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() # Placeholder for DropPath
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, # num_heads is an INTEGER here
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Ensure drop_path is a list of appropriate length if provided
        if not isinstance(drop_path, list):
            drop_path = [drop_path] * depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, # Pass the INTEGER num_heads directly
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i], # Use the correct drop_path for this block
                                 act_layer=nn.GELU,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def update_input_resolution(self, new_resolution):
        self.input_resolution = new_resolution
        for blk in self.blocks:
            blk.input_resolution = new_resolution
            if blk.shift_size > 0: # Recalculate mask if shifting is active
                blk.attn_mask = blk.calculate_mask(new_resolution).to(blk.attn_mask.device) # Move mask to correct device


class SwinIRBlockForVideo(nn.Module):
    """
    This is a simplified SwinIR-like block intended to be integrated into the
    MDTA_FOR_VIDEO / MDTA_FOR_VIDEO_Prompted blocks.
    It does not handle initial/final convolutions or upsampling, as those
    are managed by the main Restormer architecture.
    It focuses on the deep feature extraction part of SwinIR.
    """
    def __init__(self, dim, input_resolution, depths=[2], num_heads=8, window_size=8, # num_heads is the integer from MDTA
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size

        self.num_layers = len(depths)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        # Here, `num_heads` (the parameter to this init) is an INTEGER.
        # Each BasicLayer will use this same number of heads.
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=dim,
                               input_resolution=input_resolution,
                               depth=depths[i_layer],
                               num_heads=num_heads, # Pass the integer num_heads directly
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer)
            self.layers.append(layer)
        self.norm = norm_layer(dim)

    def forward(self, x):
        # x is expected to be B, H*W, C
        # The input_resolution of the BasicLayer blocks needs to be updated dynamically if image sizes change.
        H, W = self.input_resolution # This will be updated by parent module
        
        for layer in self.layers:
            layer.update_input_resolution((H, W)) # Update BasicLayer's resolution
            x = layer(x)

        x = self.norm(x)
        return x

    def update_input_resolution(self, new_resolution):
        """Allows updating input_resolution dynamically."""
        self.input_resolution = new_resolution
        # Propagate this update to its BasicLayers
        for layer in self.layers:
            layer.update_input_resolution(new_resolution)

class MeanShift(nn.Module):
    def __init__(self, rgb_range, sign=-1):
        super(MeanShift, self).__init__()
        self.rgb_range = rgb_range
        self.sub = nn.Parameter(torch.Tensor(3).cuda())
        self.add = nn.Parameter(torch.Tensor(3).cuda())
        if rgb_range == 1.:
            # For 0-1 input range, no mean shift needed.
            self.sub.data.fill_(0)
            self.add.data.fill_(0)
        else:
            # For 0-255 input range, subtract mean and add it back.
            self.sub.data = torch.FloatTensor([0.4488, 0.4371, 0.4040]).mul_(rgb_range).view(1,3,1,1)
            self.add.data = torch.FloatTensor([0.4488, 0.4371, 0.4040]).mul_(rgb_range).view(1,3,1,1)

        if sign == -1:
            self.sub = self.sub
            self.add = -self.add
        else:
            self.sub = -self.sub
            self.add = self.add

    def forward(self, x):
        return x.add(self.sub.expand_as(x)).mul(1.).add(self.add.expand_as(x))

class Restormer(nn.Module):
    def __init__(self, num_heads=8, channels=48):
        super(Restormer, self).__init__()
        self.weather_preprocessor = WeatherPreprocessor()
        self.grad_fn = GradientBranch()

        self.embed_conv = nn.Conv2d(9, channels, kernel_size=3, padding=1)
        self.embed_conv1 = nn.Conv2d(9, channels, kernel_size=3, padding=1)

        self.recurrent_down1 = DownSample(channels, channels * 2)
        self.recurrent_down2 = DownSample(channels * 2, channels * 4)
        self.recurrent_down3 = DownSample(channels * 4, channels * 8)

        self.transformer_block1 = TransformerBlock_for_Video(channels, num_heads)
        self.downsample1 = DownSample(channels, channels * 2)

        self.transformer_block2 = TransformerBlock_for_Video(channels * 2, num_heads)
        self.downsample2 = DownSample(channels * 2, channels * 4)

        self.transformer_block3 = TransformerBlock_for_Video(channels * 4, num_heads)
        self.downsample3 = DownSample(channels * 4, channels * 8)

        self.transformer_block4 = TransformerBlock_for_Video_Prompted(channels * 8, num_heads)
        self.bottleneck_temporal_fusion = BottleneckTemporalFusion(channels * 8)

        self.upsample1 = UpSample(channels * 8, channels * 4)
        self.transformer_block5 = TransformerBlock_for_Video(channels * 4, num_heads)
        self.proj_f1_lv3 = nn.Conv2d(channels * 4, channels * 4, kernel_size=1)

        self.upsample2 = UpSample(channels * 4, channels * 2)
        self.transformer_block6 = TransformerBlock_for_Video(channels * 2, num_heads)
        self.proj_f1_lv2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=1)

        self.upsample3 = UpSample(channels * 2, channels)
        self.transformer_block7 = TransformerBlock_for_Video(channels, num_heads)
        self.proj_f1_lv1 = nn.Conv2d(channels, channels, kernel_size=1)

        # FIX: Corrected channel arguments for DualPathFusion
        # dual_fusion1 combines (proj_f1_lv3(f1_lv3) -> channels*4) and (upsample_block1 -> channels*4)
        self.dual_fusion1 = DualPathFusion(channels * 4, channels * 4, channels * 4)
        # dual_fusion2 combines (proj_f1_lv2(f1_lv2) -> channels*2) and (upsample_block2 -> channels*2)
        self.dual_fusion2 = DualPathFusion(channels * 2, channels * 2, channels * 2)
        # dual_fusion3 combines (proj_f1_lv1(f1_lv1) -> channels) and (upsample_block3 -> channels)
        self.dual_fusion3 = DualPathFusion(channels, channels, channels)

        self.output_conv = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x_input):
        x_orig, recurrent_frame_orig = x_input.chunk(2, dim=0)

        x_processed = self.weather_preprocessor(x_orig)
        recurrent_frame_processed = self.weather_preprocessor(recurrent_frame_orig)

        with torch.no_grad():
            grad_feats_x = self.grad_fn(x_orig)
            grad_feats_recurrent = self.grad_fn(recurrent_frame_orig)

        x = torch.cat([x_processed, grad_feats_x], dim=1)
        f1_lv1_input = torch.cat([recurrent_frame_processed, grad_feats_recurrent], dim=1)

        fo = self.embed_conv(x)
        f1_lv1 = self.embed_conv1(f1_lv1_input)

        f1_lv2 = self.recurrent_down1(f1_lv1)
        f1_lv3 = self.recurrent_down2(f1_lv2)
        f1_lv4 = self.recurrent_down3(f1_lv3)

        transformer1 = self.transformer_block1(fo, f1_lv1)
        downsample_block1 = self.downsample1(transformer1)

        transformer2 = self.transformer_block2(downsample_block1, f1_lv2)
        downsample_block2 = self.downsample2(transformer2)

        transformer3 = self.transformer_block3(downsample_block2, f1_lv3)
        downsample_block3 = self.downsample3(transformer3)

        transformer4 = self.transformer_block4(downsample_block3, f1_lv4)
        fused_bottleneck = self.bottleneck_temporal_fusion(transformer4, f1_lv4)

        upsample_block1 = self.upsample1(fused_bottleneck)
        fused_lv3 = self.dual_fusion1(self.proj_f1_lv3(f1_lv3), upsample_block1)
        transformer5 = self.transformer_block5(fused_lv3, self.proj_f1_lv3(f1_lv3))

        upsample_block2 = self.upsample2(transformer5)
        fused_lv2 = self.dual_fusion2(self.proj_f1_lv2(f1_lv2), upsample_block2)
        transformer6 = self.transformer_block6(fused_lv2, self.proj_f1_lv2(f1_lv2))

        upsample_block3 = self.upsample3(transformer6)
        fused_lv1 = self.dual_fusion3(self.proj_f1_lv1(f1_lv1), upsample_block3)
        transformer7 = self.transformer_block7(fused_lv1, self.proj_f1_lv1(f1_lv1))

        out = self.output_conv(transformer7)
        out = torch.clamp(out, min=0, max=1)
        return out
