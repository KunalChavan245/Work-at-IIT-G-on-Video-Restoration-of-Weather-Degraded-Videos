
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
            nn.BatchNorm2d(32), # Added BatchNorm to match 32 output channels
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64), # Added BatchNorm to match 64 output channels
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

        # Calculate the correct number of channels required for offsets for each deformable conv
        self.offset_channels_3x3 = 2 * 3 * 3  # 18
        self.offset_channels_5x5 = 2 * 5 * 5  # 50
        self.offset_channels_7x7 = 2 * 7 * 7  # 98
        total_offset_channels = self.offset_channels_3x3 + self.offset_channels_5x5 + self.offset_channels_7x7 # 166

        # The offset generator now produces the correct total number of channels (166)
        self.offset_conv1 = SCConv(in_channels*2, total_offset_channels, 1, padding, dilation, 1, 2)

        # Deformable convolutions with groups=8 instead of groups=1
        self.deform_3x3 = DeformConv2d(in_channels, out_channels, 3, padding=1, groups=8)
        self.deform_5x5 = DeformConv2d(in_channels, out_channels, 5, padding=2, groups=8)
        self.deform_7x7 = DeformConv2d(in_channels, out_channels, 7, padding=3, groups=8)

        self.act_layer = act_layer
        self.pointwise = torch.nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x, y):
        # Generate a single tensor containing all offsets
        all_offsets = self.offset_conv1(torch.cat((x, y), dim=1))

        # Correctly split the offset tensor based on the required channels for each kernel size
        start = 0
        end = self.offset_channels_3x3
        offset_3x3 = all_offsets[:, start:end, :, :]

        start = end
        end += self.offset_channels_5x5
        offset_5x5 = all_offsets[:, start:end, :, :]

        start = end
        end += self.offset_channels_7x7
        offset_7x7 = all_offsets[:, start:end, :, :]

        # Apply deformable convolutions without the incorrect mask argument
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
        # padding=0 for 1x1 conv instead of padding=1
        self.conv1x1 = nn.Conv2d(channel_dim*2, channel_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(channel_dim*3, channel_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.local_prompt_extractor1 = nn.Conv2d(channel_dim, channel_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.local_prompt_extractor2 = nn.Conv2d(channel_dim, channel_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.Global_Prompt_Extractor = TransformerBlock(channel_dim, 8)

    def forward(self, x):
        x_local_1 = self.local_prompt_extractor1(x)
        x_local_2 = self.local_prompt_extractor2(x)
        x_global = self.Global_Prompt_Extractor(x)
        concat_feature = torch.cat([x_local_1, x_local_2, x_global], dim=1)
        prompt = self.conv3x3(concat_feature)
        return prompt

class DeformableWindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8, learnable_temp=None): # Added learnable_temp
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.learnable_temp = learnable_temp # Store the learnable temperature parameter

        self.qkv = nn.Linear(dim * window_size * window_size, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.window_reconstruct = nn.Linear(dim, dim * window_size * window_size)

        self.offset_conv = nn.Conv2d(dim, 2 * window_size * window_size, kernel_size=3, padding=1, bias=True)
        self.unfold = nn.Unfold(kernel_size=window_size, stride=window_size)
        self.fold = nn.Fold(output_size=(1, 1), kernel_size=window_size, stride=window_size) # dummy size for init

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
        # Apply learnable temperature here
        if self.learnable_temp is not None:
            attn = attn * self.learnable_temp # Temperature is (1, num_heads, 1, 1) and will broadcast

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
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1)) # This is the learnable temperature
        self.temporal_align = Temporal_Alignment(channels, channels, 3)
        self.norm = nn.GroupNorm(1, channels)
        # Pass the learnable temperature to DeformableWindowAttention
        self.attn = DeformableWindowAttention(dim=channels, window_size=8, num_heads=num_heads, learnable_temp=self.temperature)

    def forward(self, x, y):
        b, c, h, w = x.shape
        k = self.temporal_align(x, y)
        x_normed = self.norm(x.view(b, c, -1)).view(b, c, h, w)
        # The attention output is already scaled by temperature inside self.attn
        return x + self.attn(x_normed)

class MDTA_FOR_VIDEO_Prompted(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1)) # This is the learnable temperature
        self.prompt_block = Multi_Receptive_Attentive_Prompts(channel_dim=channels)
        self.temporal_align = Temporal_Alignment(channels, channels, 3)
        self.norm = nn.GroupNorm(1, channels * 2)
        # Pass the learnable temperature to DeformableWindowAttention
        self.attn = DeformableWindowAttention(dim=channels * 2, window_size=8, num_heads=num_heads, learnable_temp=self.temperature)
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

    def forward(self, x, y):
        b, c, h, w = x.shape
        prompt = self.prompt_block(x)
        x_prompted = torch.cat([x, prompt], dim=1)
        k = self.temporal_align(x, y) # k is not used directly in this specific return line
        x_normed = self.norm(x_prompted.view(b, c * 2, -1)).view(b, c * 2, h, w)

        attn_out = self.attn(x_normed)

        fused_output_channels = x_prompted + attn_out

        return self.fusion(fused_output_channels)


class TransformerBlock_for_Video(nn.Module):
    def __init__(self, channels, num_heads):
        super(TransformerBlock_for_Video, self).__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.attn = MDTA_FOR_VIDEO(channels, num_heads)
        self.hybrid_attn = HybridAttention(channels)

    def forward(self, x, y):
        assert x.shape[1] == y.shape[1], f"Channel mismatch: x={x.shape}, y={y.shape}"
        x = x + self.hybrid_attn(self.attn(self.norm1(x), self.norm2(y)))
        return x

class TransformerBlock_for_Video_Prompted(nn.Module):
    def __init__(self, channels, num_heads):
        super(TransformerBlock_for_Video_Prompted, self).__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.attn = MDTA_FOR_VIDEO_Prompted(channels, num_heads)
        self.hybrid_attn = HybridAttention(channels)

    def forward(self, x, y):
        assert x.shape[1] == y.shape[1], f"Channel mismatch: x={x.shape}, y={y.shape}"
        x = x + self.hybrid_attn(self.attn(x, y))
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
        self.conv = nn.Sequential(
            nn.Conv2d(low_res_ch + high_res_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, low_res_feat, high_res_feat):
        up_high = F.interpolate(high_res_feat, size=low_res_feat.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([low_res_feat, up_high], dim=1))

class Restormer(nn.Module):
    def __init__(self, num_heads=8, channels=16, expansion_factor=2.66):
        super(Restormer, self).__init__()
        self.weather_preprocessor = WeatherPreprocessor()
        self.embed_conv = nn.Conv2d(9, channels, kernel_size=3, padding=1, bias=False)
        self.embed_conv1 = nn.Conv2d(9, channels, kernel_size=3, padding=1, bias=False)

        self.transformer_block1 = TransformerBlock_for_Video_Prompted(channels, num_heads)
        self.transformer_block2 = TransformerBlock_for_Video_Prompted(channels*2, num_heads)
        self.transformer_block3 = TransformerBlock_for_Video_Prompted(channels*4, num_heads)
        self.transformer_block4 = TransformerBlock_for_Video(channels*4, num_heads)

        self.bottleneck_temporal_fusion = BottleneckTemporalFusion(channels*4)

        self.transformer_block5 = TransformerBlock_for_Video(channels*2, num_heads)
        self.transformer_block6 = TransformerBlock_for_Video(channels, num_heads)
        self.transformer_block7 = TransformerBlock_for_Video(channels, num_heads)

        self.downsample1 = DownSample(channels, channels*2)
        self.downsample2 = DownSample(channels*2, channels*4)
        self.downsample3 = DownSample(channels*4, channels*4)

        self.dual_fusion1 = DualPathFusion(channels*2, channels*4, channels*2)
        self.dual_fusion2 = DualPathFusion(channels, channels*2, channels)
        self.dual_fusion3 = DualPathFusion(channels, channels, channels)

        self.output = nn.Conv2d(channels*2, 3, kernel_size=3, padding=1, bias=False)

        self.recurrent_down1 = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)
        self.recurrent_down2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1)
        self.recurrent_down3 = nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=2, padding=1)

        self.recurrent_up1 = nn.Conv2d(channels * 4, channels * 2, kernel_size=1)
        self.recurrent_up2 = nn.Conv2d(channels * 2, channels, kernel_size=1)

        self.hat1 = HybridAttention(channels)
        self.hat2 = HybridAttention(channels*2)
        self.hat3 = HybridAttention(channels*4)

        self.proj_f1_lv3_for_block5 = nn.Conv2d(channels * 4, channels * 2, kernel_size=1)
        self.proj_f1_lv2_for_block6 = nn.Conv2d(channels * 2, channels, kernel_size=1)

        # Initialize GradientBranch once
        self.grad_fn = GradientBranch()


    def forward(self, x_input):
        x_orig, recurrent_frame_orig = x_input.chunk(2, dim=0)

        x_processed = self.weather_preprocessor(x_orig)
        recurrent_frame_processed = self.weather_preprocessor(recurrent_frame_orig)

        # Use the pre-initialized GradientBranch
        # No .eval() here as it's not a temporary instance and state should be managed by the model
        with torch.no_grad(): # Keep no_grad for features, as they are not directly optimized
            grad_feats_x = self.grad_fn(x_orig)
            grad_feats_recurrent = self.grad_fn(recurrent_frame_orig)

        x = torch.cat([x_processed, grad_feats_x], dim=1)
        f1_lv1_input = torch.cat([recurrent_frame_processed, grad_feats_recurrent], dim=1)

        fo = self.embed_conv(x)
        f1_lv1 = self.embed_conv1(f1_lv1_input)
        f1_lv2 = self.recurrent_down1(f1_lv1)
        f1_lv3 = self.recurrent_down2(f1_lv2)
        f1_lv4 = self.recurrent_down3(f1_lv3)

        transformer1 = self.transformer_block1(self.hat1(fo), self.hat1(f1_lv1))
        downsample_block1 = self.downsample1(transformer1)
        transformer2 = self.transformer_block2(self.hat2(downsample_block1), self.hat2(f1_lv2))
        downsample_block2 = self.downsample2(transformer2)
        transformer3 = self.transformer_block3(self.hat3(downsample_block2), self.hat3(f1_lv3))
        downsample_block3 = self.downsample3(transformer3)

        transformer4 = self.transformer_block4(downsample_block3, f1_lv4)
        fused_bottleneck = self.bottleneck_temporal_fusion(transformer4, f1_lv4)

        transformer5 = self.transformer_block5(self.dual_fusion1(self.recurrent_up1(f1_lv3), fused_bottleneck), self.proj_f1_lv3_for_block5(f1_lv3))

        transformer6 = self.transformer_block6(self.dual_fusion2(self.recurrent_up2(f1_lv2), transformer5), self.proj_f1_lv2_for_block6(f1_lv2))

        transformer7 = self.transformer_block7(self.dual_fusion3(f1_lv1, transformer6), f1_lv1)

        concate3 = torch.cat([fo, transformer7], dim=1)
        residual = self.output(concate3)
        out = torch.clamp(residual + x_orig, min=-1, max=1)
        return out