
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

# --- Optimized Hyperparameters from Optuna ---
OPTIMIZED_SEED = 887
OPTIMIZED_LR = 1.8470213731822586e-05
OPTIMIZED_CHANNELS = 16
OPTIMIZED_NUM_HEADS = 8
# ---------------------------------------------

def parse_args():
    desc = 'Pytorch Implementation of \'Restormer: Efficient Transformer for High-Resolution Image Restoration\''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='./Dataset/AIR_HAZE_ALL/Train/Thick/')
    parser.add_argument('--data_path_test', type=str, default='./Dataset/AIR_HAZE_ALL/Test/Thick/')
    parser.add_argument('--data_name', type=str, default='Thick', choices=['unified', 'Snow', 'underwater', 'Thick'])
    parser.add_argument('--save_path', type=str, default='result')

    # Update default values to optimized ones
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[2, 3, 3, 4],
                        help='number of transformer blocks for each level')
    # Use optimized num_heads and scale it for each level
    parser.add_argument('--num_heads', nargs='+', type=int,
                        default=[OPTIMIZED_NUM_HEADS, OPTIMIZED_NUM_HEADS*2, OPTIMIZED_NUM_HEADS*4, OPTIMIZED_NUM_HEADS*8],
                        help='number of attention heads for each level')
    # Use optimized channels and scale it for each level
    parser.add_argument('--channels', nargs='+', type=int,
                        default=[OPTIMIZED_CHANNELS, OPTIMIZED_CHANNELS*2, OPTIMIZED_CHANNELS*4, OPTIMIZED_CHANNELS*8],
                        help='number of channels for each level')
    
    parser.add_argument('--expansion_factor', type=float, default=2.66, help='factor of channel expansion for GDFN')
    parser.add_argument('--num_refinement', type=int, default=4, help='number of channels for refinement stage')
    parser.add_argument('--num_iter', type=int, default=100000, help='iterations of training') # Max iterations for full training
    parser.add_argument('--batch_size', nargs='+', type=int, default=[1, 1, 1, 1, 1, 1], help='batch size for each level')
    parser.add_argument('--patch_size', nargs='+', type=int, default=[256, 256, 256, 256, 256, 256],
                        help='patch size of each image for progressive learning')
    parser.add_argument('--lr', type=float, default=OPTIMIZED_LR, help='learning rate') # Updated default LR
    parser.add_argument('--milestone', nargs='+', type=int, default=[30000, 60000, 80000], help='milestone for changing batch size')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader') # Set to 0 for better debugging on Windows. Use >0 for faster loading on Linux.
    parser.add_argument('--seed', type=int, default=OPTIMIZED_SEED, help='random seed (-1 for no manual seed)') # Updated default seed
    # model_file is None means training stage, else means testing stage
    parser.add_argument('--model_file', type=str, default=None, help='path of pre-trained model file')
    parser.add_argument('--finetune', default=True, help='path of pre-trained model file')

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


def pad_image_needed(img, size):
    width, height = T.get_image_size(img)
    if width < size[1]:
        img = T.pad(img, [size[1] - width, 0], padding_mode='reflect')
    if height < size[0]:
        img = T.pad(img, [0, size[0] - height], padding_mode='reflect')
    return img


class RainDataset(Dataset):
    def __init__(self, data_path, data_path_test, data_name, data_type, patch_size=None, length=None):
        super().__init__()
        self.data_name, self.data_type, self.patch_size = data_name, data_type, patch_size
        self.rain_images = sorted(glob.glob('{}/*.png'.format(data_path)))
        self.rain_images_test = sorted(glob.glob('{}/*.png'.format(data_path_test)))
        # make sure the length of training and testing different
        self.num = len(self.rain_images)
        self.num_test = len(self.rain_images_test)
        self.sample_num = length if data_type == 'train' else self.num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        if self.data_type == 'train':
            image_name = os.path.basename(self.rain_images[idx % self.num])

            imag = np.array(Image.open(self.rain_images[idx % self.num]))
            r,c,ch = imag.shape
            width = c//2

            rain = T.to_tensor(imag[:,:width,:])
            norain = T.to_tensor(imag[:,width:width*2,:])

            # make sure the image could be cropped
            rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
            norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
            i, j, th, tw = RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
            rain = T.crop(rain, i, j, th, tw)
            norain = T.crop(norain, i, j, th, tw)
            
            if torch.rand(1) < 0.5:
                rain = T.hflip(rain)
                norain = T.hflip(norain)
            if torch.rand(1) < 0.5:
                rain = T.vflip(rain)
                norain = T.vflip(norain)

        else:
            image_name = os.path.basename(self.rain_images_test[idx % self.num_test])

            imag = np.array(Image.open(os.path.join(self.rain_images_test[idx % self.num_test])))
            r,c,ch = imag.shape
            width = c//2

            rain = T.to_tensor(imag[:,:width,:]) # Assuming left half is rain (input)
            norain = T.to_tensor(imag[:,width:width*2,:]) # Assuming right half is clean (ground truth)
            h, w = rain.shape[1:]
            
        return rain, norain, image_name, h, w


class VideoFrameDataset(Dataset):
    def __init__(self, folder_path, data_type, length=None):
        self.folder_path = folder_path
        self.data_type = data_type
        self.video_folders = sorted(os.listdir(folder_path))
        
        self.sample_num = length if self.data_type == 'train' else len(self.video_folders) 
        
    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx % len(self.video_folders)]
        frame_files = sorted(os.listdir(os.path.join(self.folder_path, video_folder)))
        
        frames_ip = []
        frames_gt = []
        
        if self.data_type == 'train':
            for i in range(len(frame_files)):
                frame1 = np.array(Image.open(os.path.join(self.folder_path, video_folder, frame_files[i])))
                frame1_ip = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.Resize((128, 128))(T.to_tensor(frame1[:,frame1.shape[1]//2:,:])))
                frame1_gt = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.Resize((128, 128))(T.to_tensor(frame1[:,:frame1.shape[1]//2,:])))
                
                if torch.rand(1) < 0.5:
                    frame1_ip = T.hflip(frame1_ip)
                    frame1_gt = T.hflip(frame1_gt)
                if torch.rand(1) < 0.5:
                    frame1_ip = T.vflip(frame1_ip)
                    frame1_gt = T.vflip(frame1_gt)
                    
                frames_ip.append(frame1_ip)
                frames_gt.append(frame1_gt)

        else:
            for i in range(len(frame_files)):
                frame1 = np.array(Image.open(os.path.join(self.folder_path, video_folder, frame_files[i])))
                # Assuming the image is split: left half is GT, right half is input (rainy)
                frame1_ip = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.Resize((128,128))(T.to_tensor(frame1[:,frame1.shape[1]//2:,:])))
                frame1_gt = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.Resize((128, 128))(T.to_tensor(frame1[:,:frame1.shape[1]//2,:])))
                
                frames_ip.append(frame1_ip)
                frames_gt.append(frame1_gt)
               
        return frames_ip, frames_gt


def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0) # Add 16.0 for Y in YCbCr


def psnr(x, y, data_range=255.0):
    # x and y are expected to be in range [0, data_range] for correct calculation
    # If inputs are byte (0-255), this will implicitly convert them to float for division
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return 100.0 # Perfect match
    
    
    pixel_max_for_formula = 255.0 # For the standard PSNR definition.

    return 20 * torch.log10(data_range / torch.sqrt(mse * (data_range ** 2))) # More robust to original data_range

def ssim(img1, img2, data_range=255, size_average=True, C1=0.01**2, C2=0.03**2):
    K = [C1, C2]
    
    # Ensure inputs are float for calculation
    img1 = img1.float() # Ensure float, as they might be .double() from main.py
    img2 = img2.float()

    window_size = 11
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]).to(img1.device) # Move to device
    _1D_window = gauss/gauss.sum()
    _2D_window = _1D_window.unsqueeze(0) * _1D_window.unsqueeze(1)
    window = _2D_window.expand(1, 1, window_size, window_size).contiguous().to(img1.device) # Move to device

    # img1 and img2 should already be (N, 1, H, W). No need for unsqueeze(1)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1.pow(2), window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2.pow(2), window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2

    numerator = (2 * mu1_mu2 + K[0]) * (2 * sigma12 + K[1])
    denominator = (mu1_sq + mu2_sq + K[0]) * (sigma1_sq + sigma2_sq + K[1])
    
    ssim_map = numerator / (denominator + 1e-12) # Added small epsilon for stability

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
            input = self.transform(input, mode='bilinear', size=(128, 128), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(128, 128), align_corners=False)
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

