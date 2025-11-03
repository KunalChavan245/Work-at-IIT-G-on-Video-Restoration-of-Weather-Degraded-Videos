import argparse
import glob
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as T
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
import torchvision

# Removed optimized hyperparameters, will use default values below
# --- Optimized Hyperparameters from Optuna ---
# OPTIMIZED_SEED = 887
# OPTIMIZED_LR = 1.8470213731822586e-05
# OPTIMIZED_CHANNELS = 48 # Adjusted to match the default channels of your original Restormer
# OPTIMIZED_NUM_HEADS = 8 # Keep as 8 as it's the base for num_heads lists
# ---------------------------------------------

def parse_args():
    desc = 'Pytorch Implementation of \'Restormer: Efficient Transformer for High-Resolution Image Restoration\''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='./Dataset/Train/') # Updated for video dataset structure
    parser.add_argument('--data_path_test', type=str, default='./Dataset/Test/') # Updated for video dataset structure
    parser.add_argument('--data_name', type=str, default='multi_weather_video',
                        choices=['unified', 'Snow', 'underwater', 'Thick', 'multi_weather_video'])
    parser.add_argument('--save_path', type=str, default='result')

    parser.add_argument('--num_blocks', nargs='+', type=int, default=[2, 3, 3, 4],
                        help='number of transformer blocks for each level')
    
    # Reverted to common Restormer base channel (48) for hyperparameter tuning
    parser.add_argument('--channels', nargs='+', type=int,
                        default=[48, 96, 192, 384], # Default channels scaled by 2
                        help='number of channels for each level')
    
    # Reverted to common Restormer base num_heads (8) for hyperparameter tuning
    parser.add_argument('--num_heads', nargs='+', type=int,
                        default=[8, 8, 8, 8], # Common num_heads, typically not scaled in the same way as channels
                        help='number of attention heads for each level')
    
    parser.add_argument('--expansion_factor', type=float, default=2.66, help='factor of channel expansion for GDFN')
    parser.add_argument('--num_refinement', type=int, default=4, help='number of channels for refinement stage')
    parser.add_argument('--num_iter', type=int, default=700000, help='iterations of training')
    parser.add_argument('--batch_size', nargs='+', type=int, default=[1, 1, 1, 1, 1, 1], # Batch size is 1 for video sequences
                        help='batch size for each level (number of videos)')
    parser.add_argument('--patch_size', nargs='+', type=int, default=[256, 256, 256, 256, 256, 256],
                        help='patch size of each frame for progressive learning')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') # Reverted to a common default LR for tuning
    parser.add_argument('--milestone', nargs='+', type=int, default=[150000,300000,500000], help='milestone for changing batch size')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=42, help='random seed (-1 for no manual seed)') # Reverted to a common default seed for tuning
    parser.add_argument('--model_file', type=str, default=None, help='path of pre-trained model file')
    parser.add_argument('--finetune', action='store_true', help='Set to True if fine-tuning a pre-trained model')

    args = parser.parse_args()
    return init_args(args)


class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_path_test = args.data_path_test
        self.data_name = args.data_name
        self.save_path = args.save_path
        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.channels = args.channels
        self.expansion_factor = args.expansion_factor
        self.num_refinement = args.num_refinement
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.lr = args.lr
        self.milestone = args.milestone
        self.workers = args.workers
        self.model_file = args.model_file
        self.finetune = args.finetune


def init_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return Config(args)


class VideoFrameDataset(Dataset):
    def __init__(self, folder_path, data_type, length=None):
        self.folder_path = folder_path
        self.data_type = data_type
        self.video_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
        
        self.sample_num = length if self.data_type == 'train' and length is not None else len(self.video_folders)
        
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.resize_transform = transforms.Resize((128,128))

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        video_folder_name = self.video_folders[idx % len(self.video_folders)]
        full_video_path = os.path.join(self.folder_path, video_folder_name)
        frame_files = sorted([f for f in os.listdir(full_video_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        frames_ip = []
        frames_gt = []
        
        for frame_file in frame_files:
            frame_path = os.path.join(full_video_path, frame_file)
            frame_img_pil = Image.open(frame_path).convert('RGB')
            
            width, height = frame_img_pil.size
            
            ip_frame_pil = frame_img_pil.crop((width // 2, 0, width, height))
            gt_frame_pil = frame_img_pil.crop((0, 0, width // 2, height))

            ip_tensor = T.to_tensor(ip_frame_pil)
            gt_tensor = T.to_tensor(gt_frame_pil)

            ip_tensor = self.normalize(self.resize_transform(ip_tensor))
            gt_tensor = self.normalize(self.resize_transform(gt_tensor))
            
            if self.data_type == 'train':
                if torch.rand(1) < 0.5:
                    ip_tensor = T.hflip(ip_tensor)
                    gt_tensor = T.hflip(gt_tensor)
                if torch.rand(1) < 0.5:
                    ip_tensor = T.vflip(ip_tensor)
                    gt_tensor = T.vflip(gt_tensor)
                if torch.rand(1) < 0.5:
                    angle = random.choice([90, 180, 270])
                    ip_tensor = T.rotate(ip_tensor, angle)
                    gt_tensor = T.rotate(gt_tensor, angle)
                    
            frames_ip.append(ip_tensor)
            frames_gt.append(gt_tensor)
       
        return torch.stack(frames_ip), torch.stack(frames_gt)


def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)


def psnr(x, y, data_range=255.0):
    x, y = x.float(), y.float() 
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(data_range / torch.sqrt(mse))

def ssim(img1, img2, data_range=255, size_average=True, C1=0.01**2, C2=0.03**2):
    img1 = img1.float()
    img2 = img2.float()

    window_size = 11
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]).to(img1.device)
    _1D_window = gauss/gauss.sum()
    _2D_window = _1D_window.unsqueeze(0) * _1D_window.unsqueeze(1)
    window = _2D_window.expand(1, 1, window_size, window_size).contiguous().to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1.pow(2), window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2.pow(2), window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / (denominator + 1e-12)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(128,128), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(128,128), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

