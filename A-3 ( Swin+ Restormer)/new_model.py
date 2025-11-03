import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Original SCConv and Temporal_Alignment classes
# (No changes proposed for these unless specifically mentioned in recommendations)
class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes*2, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes*2, kernel_size=3, stride=1,
                                padding=1, dilation=dilation,
                                groups=groups, bias=False),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, 18, kernel_size=3, stride=stride,
                                padding=1, dilation=dilation,
                                groups=1, bias=False),
                    )

    def forward(self, x):
        identity = x
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))
        out = torch.mul(self.k3(x), out)
        out = self.k4(out)
        return out

class Temporal_Alignment(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, # This kernel_size is not used in the original DeformConv2d init, but kept for signature.
                 stride=1,
                 padding=0,
                 dilation=1,
                 act_layer=nn.ReLU()):
        super(Temporal_Alignment, self).__init__()
        self.offset_conv1 = SCConv(in_channels*2, out_channels, 1, padding, dilation, 1, 2)
        # DeformConv2d kernel_size should be consistent with input. Original code uses 3.
        # Ensure padding is correctly applied for kernel_size=3, padding=1.
        self.deform1 =  DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=8)
        self.act_layer = act_layer
        self.pointwise = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1) # Changed in_channels to out_channels as deform1 outputs out_channels

    def offset_gen(self, x, y):
        offset = torch.cat((x, y), dim=1)
        offset = self.offset_conv1(offset)
        mask = torch.sigmoid(offset)
        return offset, mask

    def forward(self, x, y):
        offset1, mask = self.offset_gen(x, y)
        feat1 = self.deform1(x, offset1, mask[:,:mask.shape[1]//2,:,:])
        x = self.act_layer(feat1)
        x = self.pointwise(x)
        return x

# New: Weather Prior Guided Module (WPGM)
class WeatherPriorGuidedModule(nn.Module):
    def __init__(self, channels):
        super(WeatherPriorGuidedModule, self).__init__()
        # Simplified WPGM: learns to extract and use a "prior" to modulate features.
        # In a real scenario, this would involve querying a learned bank of priors
        # or using a small network to predict prior maps.
        self.prior_extractor = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid() # To produce a modulation mask/prior map
        )
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        prior_map = self.prior_extractor(x)
        # Simple modulation: element-wise multiplication with the prior map
        # A more sophisticated WPGM might use FiLM layers or adaptive instance normalization
        # based on the prior.
        modulated_x = x * prior_map
        return self.conv(modulated_x) + x # Residual connection

# Original Multi_Receptive_Attentive_Prompts, MDTA, TransformerBlock
class Multi_Receptive_Attentive_Prompts(nn.Module):
    def __init__(self,channel_dim=32):
        super(Multi_Receptive_Attentive_Prompts,self).__init__()
        self.conv3x3 = nn.Conv2d(channel_dim*3,channel_dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.local_prompt_extractor1 = nn.Conv2d(channel_dim,channel_dim,kernel_size=1,stride=1,padding=0,bias=False)
        self.local_prompt_extractor2 = nn.Conv2d(channel_dim,channel_dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.Global_Prompt_Extractor = TransformerBlock(channel_dim, 8) # Assuming TransformerBlock is defined later or imported

    def forward(self,x):
        x_local_1 = self.local_prompt_extractor1(x)
        x_local_2 = self.local_prompt_extractor2(x)
        x_global = self.Global_Prompt_Extractor(x)
        concat_feature = torch.cat([x_local_1, x_local_2, x_global], dim=1)
        prompt = self.conv3x3(concat_feature)
        return prompt

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        # self.norm2 = nn.LayerNorm(channels) # Commented out in original
        # self.ffn = GDFN(channels, expansion_factor) # Commented out in original

    def forward(self, x):
        b, c, h, w = x.shape
        # Permute to (B, H*W, C) for LayerNorm, then back
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        return x

# Modified MDTA_FOR_VIDEO for enhanced Temporal Consistency (simplified Dynamic Routing)
class MDTA_FOR_VIDEO(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA_FOR_VIDEO, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        # Modified qkv to generate query for current frame and key/value for aligned prev frame
        # q_curr for current frame, kv_prev for aligned previous frame
        self.q_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.kv_conv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False) # For K and V from aligned frame
        
        self.q_conv_depth = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.kv_conv_depth = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
        
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.Temporal_Alignment_Block = Temporal_Alignment(channels, channels, 3)

    def forward(self, x, y): # x is current frame feature, y is recurrent frame feature
        b, c, h, w = x.shape
        
        # Current frame query
        q = self.q_conv_depth(self.q_conv(x))

        # Align previous frame to current frame's perspective
        aligned_y = self.Temporal_Alignment_Block(y, x) # Align y (recurrent) to x (current)
        
        # Generate key and value from the aligned previous frame
        k, v = self.kv_conv_depth(self.kv_conv(aligned_y)).chunk(2, axis=1)
        
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out

class MDTA_FOR_VIDEO_Prompted(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA_FOR_VIDEO_Prompted, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False) # Changed from channels*2 as prompt is concated to query, not value
        self.prompting_block = Multi_Receptive_Attentive_Prompts(channel_dim=channels)
        self.Temporal_Alignment_Block = Temporal_Alignment(channels, channels, 3)

        # New convolutions for QKV from prompted/aligned features
        self.q_proj = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False) # Query from x + prompt
        self.kv_proj = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False) # Key/Value from aligned y

    def forward(self, x, y):
        b, c, h, w = x.shape
        
        # Generate prompt for current frame and concatenate to form query
        prompt = self.prompting_block(x)
        q_with_prompt = torch.cat([x, prompt], dim=1)
        q = self.q_proj(q_with_prompt) # Project concatenated feature to original channel dim for query
        
        # Align previous frame (y) to current frame (x)
        aligned_y = self.Temporal_Alignment_Block(y, x)
        k, v = self.kv_proj(aligned_y).chunk(2, axis=1) # Keys and values from aligned y

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class TransformerBlock_for_Video_Prompted(nn.Module):
    def __init__(self, channels, num_heads):
        super(TransformerBlock_for_Video_Prompted, self).__init__()
        self.norm1 = nn.LayerNorm(channels) # For current frame x
        self.norm2 = nn.LayerNorm(channels) # For recurrent frame y
        self.attn = MDTA_FOR_VIDEO_Prompted(channels, num_heads)

    def forward(self, x, y):
        b, c, h, w = x.shape
        b1, c1, h1, w1 = y.shape # Should be same as x usually, but using separate for robustness
        
        # Apply LayerNorm on permuted tensors, then revert for attention
        normed_x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # (B, C, H, W) -> (B, H, W, C) -> LayerNorm -> (B, C, H, W)
        normed_y = self.norm2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x = x + self.attn(normed_x, normed_y)
        return x

class TransformerBlock_for_Video(nn.Module):
    def __init__(self, channels, num_heads):
        super(TransformerBlock_for_Video, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.attn = MDTA_FOR_VIDEO(channels, num_heads)

    def forward(self, x, y):
        b, c, h, w = x.shape
        b1, c1, h1, w1 = y.shape
        
        normed_x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        normed_y = self.norm2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x = x + self.attn(normed_x, normed_y)
        return x

# New: Mamba-like block for efficiency
class MambaBlock(nn.Module):
    def __init__(self, channels, causal_conv_kernel_size=3):
        super(MambaBlock, self).__init__()
        # Simplified Mamba-like block. A full Mamba (S4/S6) implementation is complex.
        # This conceptually captures the idea of sequence modeling (spatial sequence here)
        # with linear complexity.
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.conv_spatial = nn.Conv2d(channels, channels, kernel_size=causal_conv_kernel_size, 
                                      padding=causal_conv_kernel_size // 2, groups=channels) # Depthwise
        self.gate = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        identity = x
        
        # Reshape for sequence processing (e.g., scan lines) - conceptually
        # For true Mamba, this would involve more sophisticated state-space operations.
        # Here, we simulate a simple gated spatial convolution.
        
        # Apply layer norm on (B, H*W, C) or (B, C, H, W)
        normed_x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Gated mechanism inspired by Mamba's input projection and gating
        hidden = self.conv1(normed_x)
        x_proj, gate_proj = hidden.chunk(2, dim=1) # Split into data path and gate path

        spatial_feat = self.conv_spatial(x_proj)
        gate = torch.sigmoid(self.gate(gate_proj))
        
        out = spatial_feat * gate
        return identity + self.proj_out(out) # Residual connection

# New: Frequency Domain Processing Blocks (placeholders)
class WaveletMambaBlock(nn.Module):
    def __init__(self, channels):
        super(WaveletMambaBlock, self).__init__()
        # Placeholder for Wavelet-based Mamba Block.
        # This would typically involve:
        # 1. Wavelet transform (DWT)
        # 2. Applying Mamba-like processing on subbands
        # 3. Inverse Wavelet transform (IDWT)
        self.mamba_block = MambaBlock(channels) # Use the simplified Mamba block
        self.conv = nn.Conv2d(channels, channels, kernel_size=1) # For channel adjustment if needed
        # In a full implementation, DWT/IDWT layers would be here.
        # For now, just pass through Mamba block.
    
    def forward(self, x):
        # Conceptual: Apply wavelet transform, process subbands, then inverse.
        # For simplicity, apply Mamba block directly.
        return self.conv(self.mamba_block(x)) + x # Residual connection

class FastFourierAdjustmentBlock(nn.Module):
    def __init__(self, channels):
        super(FastFourierAdjustmentBlock, self).__init__()
        # Placeholder for Fast Fourier Adjustment Block.
        # This would typically involve:
        # 1. FFT
        # 2. Frequency domain manipulation (e.g., learned filters, magnitude/phase adjustment)
        # 3. IFFT
        self.conv_spatial = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_freq_transform = nn.Conv2d(channels, channels, kernel_size=1)
        # Actual FFT/IFFT operations and frequency domain manipulation would go here.
        
    def forward(self, x):
        identity = x
        # Conceptual: Apply FFT, manipulate in frequency domain, then IFFT.
        # For simplicity, simulate frequency-aware processing with a convolution and
        # a learned transformation.
        
        # A true FFAB would perform:
        # x_fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        # # Manipulate x_fft (e.g., apply frequency-specific weights)
        # x_processed_fft = self.learnable_freq_filter(x_fft)
        # x_spatial = torch.fft.ifft2(x_processed_fft, dim=(-2, -1), norm="ortho").real
        
        # Here, a simplified approximation:
        spatial_processed = self.conv_spatial(x)
        freq_info_extracted = self.conv_freq_transform(spatial_processed)
        
        return identity + freq_info_extracted # Residual connection (simplified)

# Original DownSample and UpSample
class DownSample(nn.Module):
    def __init__(self, channels,channels_out):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels_out, kernel_size=3, padding=1, bias=False, stride = 2),
                                  )
    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor = 2):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.gated_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1, padding = 1, dilation=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x

# Modified Restormer to integrate new modules
class Restormer(nn.Module):
    def __init__(self, num_heads=8, channels=16,
                 expansion_factor=2.66): # expansion_factor not used currently, kept for compatibility
        super(Restormer, self).__init__()

        self.embed_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.embed_conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        
        # New: Weather Prior Guided Module
        self.wpgm = WeatherPriorGuidedModule(channels) # Integrated after initial embedding

        # Transformer blocks as before, some could be replaced by Mamba later
        self.transformer_block1 = TransformerBlock_for_Video_Prompted(channels, num_heads)
        self.transformer_block2 = TransformerBlock_for_Video_Prompted(channels*2, num_heads)
        # Replacing transformer_block3 with a Mamba-based block for efficiency
        # self.transformer_block3 = TransformerBlock_for_Video_Prompted(channels*4, num_heads)
        self.mamba_block3 = MambaBlock(channels*4) # New Mamba block

        self.transformer_block4 = TransformerBlock_for_Video(channels*4, num_heads)
        
        # Adding frequency domain blocks
        self.ffab1 = FastFourierAdjustmentBlock(channels * 4) # Before downsample
        self.wmb1 = WaveletMambaBlock(channels * 4) # After downsample

        self.transformer_block5 = TransformerBlock_for_Video(channels*2, num_heads)
        self.transformer_block6 = TransformerBlock_for_Video(channels, num_heads)
        self.transformer_block7 = TransformerBlock_for_Video(channels, num_heads)

        self.downsample1 = DownSample(channels, channels*2)
        self.downsample2 = DownSample(channels*2, channels*4)
        self.downsample3 = DownSample(channels*4, channels*4)

        self.upsample1 = UpSample(channels*4, channels*2)
        # Correcting input channels for upsample2, it takes concat of transformer5 and transformer3
        # transformer3 is now mamba_block3 output (channels*4), transformer5 is channels*2
        # So concat is channels*2 + channels*4 = channels*6. This remains correct.
        self.upsample2 = UpSample(channels*6, channels)
        # Correcting input channels for upsample3, it takes concat of transformer6 and transformer2
        # transformer6 is channels, transformer2 is channels*2
        # So concat is channels + channels*2 = channels*3. This remains correct.
        self.upsample3 = UpSample(channels*3, channels)
        
        # Recurrent downsample blocks remain the same
        self.recurrent_downsample1 = DownSample(channels, channels*2)
        self.recurrent_downsample2_1 = DownSample(channels, channels*2)
        self.recurrent_downsample2_2 = DownSample(channels*2, channels*4)
        self.recurrent_downsample3_1 = DownSample(channels, channels)
        self.recurrent_downsample3_2 = DownSample(channels, channels)
        self.recurrent_downsample3_3 = DownSample(channels, channels*4)
        self.recurrent_downsample4_1 = DownSample(channels, channels)
        self.recurrent_downsample4_2 = DownSample(channels, channels*2)
        self.recurrent_downsample5_1 = DownSample(channels, channels)

        self.output = nn.Conv2d(channels*2, 3, kernel_size=3, padding=1, bias=False)
        self.tan = nn.Tanh()

    def forward(self, x):
        x, recurrent_frame = x.chunk(2, dim=0) # Assuming batch is [current_frame, recurrent_frame]

        fo = self.embed_conv(x)
        f1 = self.embed_conv1(recurrent_frame)
        
        # Apply Weather Prior Guided Module
        fo = self.wpgm(fo) 
        f1 = self.wpgm(f1) # Apply WPGM to recurrent frame features as well

        transformer1 = self.transformer_block1(fo, f1)
        downsample_block1 = self.downsample1(transformer1) # 128
        
        transformer2 = self.transformer_block2(downsample_block1, self.recurrent_downsample1(f1))
        downsample_block2 = self.downsample2(transformer2) # 64
        
        # Using Mamba block instead of TransformerBlock3 for efficiency (Rec. 5)
        # original: transformer3 = self.transformer_block3(downsample_block2, self.recurrent_downsample2_2(self.recurrent_downsample2_1(f1)))
        
        # Apply Frequency Domain blocks (Rec. 6)
        freq_adj_feat_before_mamba = self.ffab1(downsample_block2)
        mamba_input_feat = freq_adj_feat_before_mamba + self.recurrent_downsample2_2(self.recurrent_downsample2_1(f1))
        transformer3 = self.mamba_block3(mamba_input_feat) # Apply Mamba block
        
        downsample_block3 = self.downsample3(transformer3) # 32
        
        # Apply another frequency domain block after downsampling
        wavelet_mamba_feat = self.wmb1(downsample_block3)
        transformer4 = self.transformer_block4(wavelet_mamba_feat, self.recurrent_downsample3_3(self.recurrent_downsample3_2(self.recurrent_downsample3_1(f1))))
        
        upsample_block1 = self.upsample1(transformer4) # 64

        transformer5 = self.transformer_block5(upsample_block1, self.recurrent_downsample4_2(self.recurrent_downsample4_1(f1)))
        concate1 = torch.cat([transformer5,transformer3], axis = 1)
        upsample_block2 = self.upsample2(concate1) # 128
        
        transformer6 = self.transformer_block6(upsample_block2, self.recurrent_downsample5_1(f1))
        concate2 = torch.cat([transformer6,transformer2], axis = 1)
        upsample_block3 = self.upsample3(concate2) # 256
        
        transformer7 = self.transformer_block7(upsample_block3, f1)
        concate3 = torch.cat([fo,transformer7], axis = 1)

        out = self.tan(self.output(concate3)+x)
        return out