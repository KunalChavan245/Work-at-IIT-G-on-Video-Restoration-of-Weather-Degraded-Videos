import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import torchvision
import matplotlib.pyplot as plt
import numpy as np

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
        
        # FIX: Calculate the correct number of channels required for offsets for each deformable conv
        self.offset_channels_3x3 = 2 * 3 * 3  # 18
        self.offset_channels_5x5 = 2 * 5 * 5  # 50
        self.offset_channels_7x7 = 2 * 7 * 7  # 98
        total_offset_channels = self.offset_channels_3x3 + self.offset_channels_5x5 + self.offset_channels_7x7 # 166

        # FIX: The offset generator now produces the correct total number of channels (166)
        self.offset_conv1 = SCConv(in_channels*2, total_offset_channels, 1, padding, dilation, 1, 2)
        
        # Deformable convolutions remain the same
        self.deform_3x3 = DeformConv2d(in_channels, out_channels, 3, padding=1, groups=8)
        self.deform_5x5 = DeformConv2d(in_channels, out_channels, 5, padding=2, groups=8)
        self.deform_7x7 = DeformConv2d(in_channels, out_channels, 7, padding=3, groups=8)
        
        self.act_layer = act_layer
        self.pointwise = torch.nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x, y):
        # Generate a single tensor containing all offsets
        all_offsets = self.offset_conv1(torch.cat((x, y), dim=1))
        
        # FIX: Correctly split the offset tensor based on the required channels for each kernel size
        start = 0
        end = self.offset_channels_3x3
        offset_3x3 = all_offsets[:, start:end, :, :]

        start = end
        end += self.offset_channels_5x5
        offset_5x5 = all_offsets[:, start:end, :, :]

        start = end
        end += self.offset_channels_7x7
        offset_7x7 = all_offsets[:, start:end, :, :]
        
        # FIX: Apply deformable convolutions without the incorrect mask argument
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
    def __init__(self,channel_dim=32):
        super(Multi_Receptive_Attentive_Prompts,self).__init__()
        self.conv1x1 = nn.Conv2d(channel_dim*2,channel_dim,kernel_size=1,stride=1,padding=1,bias=False)
        self.conv3x3 = nn.Conv2d(channel_dim*3,channel_dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.local_prompt_extractor1 = nn.Conv2d(channel_dim,channel_dim,kernel_size=1,stride=1,padding=0,bias=False)
        self.local_prompt_extractor2 = nn.Conv2d(channel_dim,channel_dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.Global_Prompt_Extractor = TransformerBlock(channel_dim, 8)
        
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

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        return x

class MDTA_FOR_VIDEO(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA_FOR_VIDEO, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.Temporal_Alignment_Block = Temporal_Alignment(channels, channels, 3)

    def forward(self, x, y):
        b, c, h, w = x.shape
        qkv_con = self.qkv(x)
        qkv_con = self.qkv_conv(qkv_con)
        q,v = qkv_con.chunk(2, axis = 1)
        k = self.Temporal_Alignment_Block(q,y)
        
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

        self.project_out = nn.Conv2d(channels*2, channels, kernel_size=1, bias=False)
        self.prompting_block = Multi_Receptive_Attentive_Prompts(channel_dim=channels)
        self.Temporal_Alignment_Block = Temporal_Alignment(channels, channels, 3)

    def forward(self, x, y):
        b, c, h, w = x.shape
        # FIX: Concatenate along the channel dimension (dim=1)
        q = torch.cat([x, self.prompting_block(x)], dim=1)
        k = self.Temporal_Alignment_Block(x,y)
        v = x

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

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA_FOR_VIDEO_Prompted(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x, y):
        b, c, h, w = x.shape
        b1, c1, h1, w1 = y.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w), self.norm2(y.reshape(b1, c1, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b1, c1, h1, w1))
        return x

class TransformerBlock_for_Video(nn.Module):
    def __init__(self, channels, num_heads):
        super(TransformerBlock_for_Video, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA_FOR_VIDEO(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x, y):
        b, c, h, w = x.shape
        b1, c1, h1, w1 = y.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w), self.norm2(y.reshape(b1, c1, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b1, c1, h1, w1))
        return x
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

class Restormer(nn.Module):
    def __init__(self, num_heads=8, channels=16,
                 expansion_factor=2.66):
        super(Restormer, self).__init__()

        self.embed_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.embed_conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.transformer_block1 = TransformerBlock_for_Video_Prompted(channels, num_heads)
        self.transformer_block2 = TransformerBlock_for_Video_Prompted(channels*2, num_heads)
        self.transformer_block3 = TransformerBlock_for_Video_Prompted(channels*4, num_heads)
        self.transformer_block4 = TransformerBlock_for_Video(channels*4, num_heads)
        self.transformer_block5 = TransformerBlock_for_Video(channels*2, num_heads)
        self.transformer_block6 = TransformerBlock_for_Video(channels, num_heads)
        self.transformer_block7 = TransformerBlock_for_Video(channels, num_heads)

        self.downsample1 = DownSample(channels, channels*2)
        self.downsample2 = DownSample(channels*2, channels*4)
        self.downsample3 = DownSample(channels*4, channels*4)

        self.upsample1 = UpSample(channels*4, channels*2)
        self.upsample2 = UpSample(channels*6, channels)
        self.upsample3 = UpSample(channels*3, channels)
        
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
        # FIX: Use more descriptive variable names for clarity
        x_current, recurrent_frame = x.chunk(2, dim=0)
        
        fo = self.embed_conv(x_current)
        f1 = self.embed_conv1(recurrent_frame)
        
        transformer1 = self.transformer_block1(fo,f1)
        downsample_block1 = self.downsample1(transformer1)
        transformer2 = self.transformer_block2(downsample_block1, self.recurrent_downsample1(f1))
        downsample_block2 = self.downsample2(transformer2)
        transformer3 = self.transformer_block3(downsample_block2, self.recurrent_downsample2_2(self.recurrent_downsample2_1(f1)))
        downsample_block3 = self.downsample3(transformer3)
        transformer4 = self.transformer_block4(downsample_block3, self.recurrent_downsample3_3(self.recurrent_downsample3_2(self.recurrent_downsample3_1(f1))))
        upsample_block1 = self.upsample1(transformer4)

        transformer5 = self.transformer_block5(upsample_block1, self.recurrent_downsample4_2(self.recurrent_downsample4_1(f1)))
        concate1 = torch.cat([transformer5,transformer3], axis = 1)
        upsample_block2 = self.upsample2(concate1)
        transformer6 = self.transformer_block6(upsample_block2, self.recurrent_downsample5_1(f1))
        concate2 = torch.cat([transformer6,transformer2], axis = 1)
        upsample_block3 = self.upsample3(concate2)
        transformer7 = self.transformer_block7(upsample_block3, f1)
        concate3 = torch.cat([fo,transformer7], axis = 1)

        # The residual connection adds the original current frame to the output
        out = self.tan(self.output(concate3) + x_current)
        return out
    