import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# def visualize_featuremaps_batchwise(ylh, filename):
#   processed = []
#   for feature_map in ylh:
#       feature_map = feature_map.squeeze(0)
#       gray_scale = feature_map[0]
#       gray_scale = gray_scale / feature_map.shape[0]
#       processed.append(gray_scale.data.cpu().numpy())

#   # fig = plt.figure(figsize=(10, 20))
#   # for i in range(len(processed)):
#   #   a = fig.add_subplot(1, 2, i+1)
#   #   imgplot = plt.imshow(processed[i])
#   #   a.axis("off")
#   #     # a.set_title(names[i].split('(')[0], fontsize=30)
#   # plt.savefig(filename, bbox_inches='tight')

#   plt.imsave(filename,processed[0])
  # plt.imshow(processed[0],cmap='Reds')
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

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        # print(self.k3(x).shape, out.shape)
        # exit(0)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out
class Temporal_Alignment(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU()):
        super(Temporal_Alignment, self).__init__()
        
        self.offset_conv1 = SCConv(in_channels*2, out_channels, 1, padding, dilation, 1, 2)
        self.deform1 =  DeformConv2d(in_channels, out_channels, 3, padding=1, groups=8)
        self.act_layer = act_layer
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def offset_gen(self, x, y):
        
        offset = torch.cat((x, y), dim=1)

        offset = self.offset_conv1(offset)
        mask = torch.sigmoid(offset)
        return offset, mask

    def forward(self, x, y):

        offset1,mask = self.offset_gen(x, y)

        # print(x.shape)
        # print(offset1.shape)
        # print(mask.shape)
        # exit(0)
        feat1 = self.deform1(x, offset1, mask[:,:mask.shape[1]//2,:,:])

        # x = self.depthwise(feat)
        x = self.act_layer(feat1)
        
        x = self.pointwise(x)
        return x


# class GDFN(nn.Module):
#     def __init__(self, channels, expansion_factor):
#         super(GDFN, self).__init__()

#         hidden_channels = int(channels * expansion_factor)
#         self.project_in = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)

#         self.conv1_1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1,
#                               groups=hidden_channels, bias=False)

#         self.conv3_3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size= 3, padding=1,
#                               groups=hidden_channels, bias=False)


#         self.conv5_5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size= 5, padding=2,
#                               groups=hidden_channels, bias=False)

#         self.project_out = nn.Conv2d(hidden_channels*2, channels, kernel_size=1, bias=False)
        


#     def forward(self, x):
#         x = self.project_in(x)
#         x1 = self.conv1_1(x)
#         x3 = self.conv3_3(x)
#         x5 = self.conv5_5(x)

#         gate1 = F.gelu(x1)*x3
#         gate2 = F.gelu(x1)*x5

#         # gate_only_1x1 = F.gelu(x1)*x1
#         # visualize_featuremaps_batchwise(gate_only_1x1, "gate_only_1x1.png")
#         # gate_only_3x3 = F.gelu(x3)*x3
#         # visualize_featuremaps_batchwise(gate_only_3x3, "gate_only_3x3.png")
#         # gate_only_5x5 = F.gelu(x5)*x5
#         # visualize_featuremaps_batchwise(gate_only_5x5, "gate_only_5x5.png")



#         x = self.project_out(torch.cat([gate1,gate2],axis = 1))
#         # visualize_featuremaps_batchwise(x, "actual.png")
#         # x1 =  self.project_out((torch.cat([gate_only_1x1,gate_only_1x1],axis = 1)))
#         # visualize_featuremaps_batchwise(x1, "x1gate_only_1x1.png")
#         # x2 =  self.project_out((torch.cat([gate_only_3x3,gate_only_3x3],axis = 1)))
#         # visualize_featuremaps_batchwise(x2, "x2gate_only_3x3.png")

#         # x3 = self.project_in(gate_only_5x5)
#         # x3 =  self.project_out((torch.cat([gate_only_5x5,gate_only_5x5],axis = 1)))
#         # visualize_featuremaps_batchwise(x3, "x3gate_only_5x5.png")



#         # x1, x3 = self.conv1(self.project_in(x)).chunk(2, dim=1)
#         # x2, x4 = self.conv2(self.project_in(x)).chunk(2, dim=1)



#         # x = self.project_out(F.gelu(x1) * x2)
#         return x

class Multi_Receptive_Attentive_Prompts(nn.Module):
    def __init__(self,channel_dim=32):
        super(Multi_Receptive_Attentive_Prompts,self).__init__()
        # self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,channel_dim,prompt_size,prompt_size))
        # self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(channel_dim*3,channel_dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.local_prompt_extractor1 = nn.Conv2d(channel_dim,channel_dim,kernel_size=1,stride=1,padding=0,bias=False)
        self.local_prompt_extractor2 = nn.Conv2d(channel_dim,channel_dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.Global_Prompt_Extractor = TransformerBlock(channel_dim, 8)
        
    def forward(self,x):
        x_local_1 = self.local_prompt_extractor1(x)
        x_local_2 = self.local_prompt_extractor2(x)
        x_global = self.Global_Prompt_Extractor(x)
        concat_feature = torch.cat([x_local_1, x_local_2, x_global], dim=1)
        # B,C,H,W = x.shape

        # emb = x.mean(dim=(-2,-1))
        
        # prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        # 
        # prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        
        # prompt = torch.sum(prompt,dim=1)
       
        # prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        
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
        # self.norm2 = nn.LayerNorm(channels)
        # self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        # x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
        #                  .contiguous().reshape(b, c, h, w))
        return x

class MDTA_FOR_VIDEO(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA_FOR_VIDEO, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # self.prompting_block = Multi_Receptive_Attentive_Prompts(channel_dim=channels,prompt_len=5,prompt_size = channels*2,lin_dim = channels)
        self.Temporal_Alignment_Block = Temporal_Alignment(channels, channels, 3)


    def forward(self, x, y):
        b, c, h, w = x.shape
        qkv_con = self.qkv(x)
        qkv_con = self.qkv_conv(qkv_con)
        # print(qkv_con.shape)
        q,v = qkv_con.chunk(2, axis = 1)
        # print(q.shape,v.shape)
        # exit(0)
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

        # self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        # self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels*2, channels, kernel_size=1, bias=False)

        self.prompting_block = Multi_Receptive_Attentive_Prompts(channel_dim=channels)
        self.Temporal_Alignment_Block = Temporal_Alignment(channels, channels, 3)


    def forward(self, x, y):
        b, c, h, w = x.shape
        q = torch.cat([x,self.prompting_block(x)])
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
        # self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x, y):
        b, c, h, w = x.shape
        b1, c1, h1, w1 = y.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w), self.norm2(y.reshape(b1, c1, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b1, c1, h1, w1))
        # x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
        #                  .contiguous().reshape(b, c, h, w))
        return x

class TransformerBlock_for_Video(nn.Module):
    def __init__(self, channels, num_heads):
        super(TransformerBlock_for_Video, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA_FOR_VIDEO(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        # self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x, y):
        b, c, h, w = x.shape
        b1, c1, h1, w1 = y.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w), self.norm2(y.reshape(b1, c1, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b1, c1, h1, w1))
        # x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
        #                  .contiguous().reshape(b, c, h, w))
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
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1, padding = 1, dilation=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x
# class UpSample(nn.Module):
#     def __init__(self, channels,channels_out):
#         super(UpSample, self).__init__()
#         self.body = nn.Sequential(nn.ConvTranspose2d(channels, channels_out, kernel_size=2, bias=False, stride = 2),
#                                  )

#     def forward(self, x):
#         return self.body(x)

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
        x, recurrent_frame = x.chunk(2, dim=0)
        
        # print(x.shape, recurrent_frame.shape)
        fo = self.embed_conv(x)

        f1 = self.embed_conv1(recurrent_frame)
        
        transformer1 = self.transformer_block1(fo,f1)
        downsample_block1 = self.downsample1(transformer1)#128
        transformer2 = self.transformer_block2(downsample_block1, self.recurrent_downsample1(f1))
        downsample_block2 = self.downsample2(transformer2)#64
        transformer3 = self.transformer_block3(downsample_block2, self.recurrent_downsample2_2(self.recurrent_downsample2_1(f1)))
        downsample_block3 = self.downsample3(transformer3)#32
        transformer4 = self.transformer_block4(downsample_block3, self.recurrent_downsample3_3(self.recurrent_downsample3_2(self.recurrent_downsample3_1(f1))))
        upsample_block1 = self.upsample1(transformer4)#64

        transformer5 = self.transformer_block5(upsample_block1, self.recurrent_downsample4_2(self.recurrent_downsample4_1(f1)))
        concate1 = torch.cat([transformer5,transformer3], axis = 1)
        upsample_block2 = self.upsample2(concate1)#128
        transformer6 = self.transformer_block6(upsample_block2, self.recurrent_downsample5_1(f1))
        concate2 = torch.cat([transformer6,transformer2], axis = 1)
        upsample_block3 = self.upsample3(concate2)#256
        transformer7 = self.transformer_block7(upsample_block3, f1)
        concate3 = torch.cat([fo,transformer7], axis = 1)



        # print(offset1.shape)
        # exit(0)
        
        # x = self.depthwise(feat)
        out = self.tan(self.output(concate3)+x)
        return out




        



        
    
