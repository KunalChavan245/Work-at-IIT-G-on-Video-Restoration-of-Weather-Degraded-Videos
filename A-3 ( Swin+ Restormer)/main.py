import os
from collections import deque
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

# Import your model and utils functions
from model import GradientBranch, Restormer
from utils import parse_args, VideoFrameDataset, rgb_to_y, psnr, ssim, VGGPerceptualLoss

# Seeding will now be handled by utils.py based on args.seed

# Global instances for perceptual loss and gradient branch
perceptual_loss = VGGPerceptualLoss().cuda()
grad_fn_for_loss = GradientBranch().cuda()

def compute_loss(pred, gt, lambda_grad=0.1):
    grad_pred = grad_fn_for_loss(pred)
    grad_gt = grad_fn_for_loss(gt)
    loss_rgb = F.l1_loss(pred, gt)
    loss_grad = F.l1_loss(grad_pred, grad_gt)
    return loss_rgb + lambda_grad * loss_grad

def charbonnier_loss(x, y, eps=1e-3):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps ** 2))

def save_loop(net, data_loader, num_iter, args, optimizer, lr_scheduler):
    global best_psnr, best_ssim, results
    val_psnr, val_ssim = test_loop_main_func(net, data_loader, num_iter, args)
    results['PSNR'].append(f'{val_psnr:.2f}')
    results['SSIM'].append(f'{val_ssim:.3f}')

    if val_psnr + val_ssim > best_psnr + best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open(f'{args.save_path}/{args.data_name}.txt', 'w') as f:
            f.write(f'Iter: {num_iter} PSNR:{best_psnr:.2f} SSIM:{best_ssim:.3f}')
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'iter': num_iter,
            'best_psnr': best_psnr,
            'best_ssim': best_ssim
        }, f'{args.save_path}/{args.data_name}.pth')

def test_loop_main_func(net, data_loader, num_iter, args):
    net.eval()
    iterator = 0
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    data_loader_iter = iter(data_loader)
    with torch.no_grad():
        test_bar = tqdm(data_loader_iter, initial=1, dynamic_ncols=True, desc=f"Test Iter [{num_iter}]")

        for rain_video_batch, norain_video_batch in test_bar:
            iterator += 1
            rain_video = rain_video_batch.squeeze(0)
            norain_video = norain_video_batch.squeeze(0)

            recurrent_frames = deque(maxlen=3)

            for k in range(rain_video.shape[0]):
                rain_frame = rain_video[k].cuda()
                norain_frame = norain_video[k].cuda()

                if k == 0:
                    input_tensor = torch.cat([rain_frame.unsqueeze(0), rain_frame.unsqueeze(0)], dim=0)
                elif len(recurrent_frames) < 3:
                    avg_recurrent = torch.mean(torch.stack(list(recurrent_frames)), dim=0)
                    input_tensor = torch.cat([rain_frame.unsqueeze(0), avg_recurrent.detach().unsqueeze(0)], dim=0)
                else:
                    avg_recurrent = torch.mean(torch.stack(list(recurrent_frames)), dim=0)
                    input_tensor = torch.cat([rain_frame.unsqueeze(0), avg_recurrent.detach().unsqueeze(0)], dim=0)

                out = net(input_tensor)
                recurrent_frames.append(out.detach().squeeze(0))

                out_display = torch.clamp(out.squeeze(0).mul(255), 0, 255).byte()
                norain_display = torch.clamp(norain_frame.mul(255), 0, 255).byte()
                rain_display = torch.clamp(rain_frame.mul(255), 0, 255).byte()
                
                # Ensure output_y and target_y are correctly named and used
                output_y_channel = rgb_to_y(out_display.double()) # This returns [1, 1, H, W]
                target_y_channel = rgb_to_y(norain_display.double()) # This returns [1, 1, H, W]
                
                # FIX: Remove .unsqueeze(0) as output_y_channel/target_y_channel are already 4D
                current_psnr = psnr(output_y_channel, target_y_channel).item()
                current_ssim = ssim(output_y_channel, target_y_channel).item()

                total_psnr += current_psnr
                total_ssim += current_ssim
                count += 1
                
                if args and args.save_path:
                    save_dir_frame = os.path.join(args.save_path, args.data_name, f"iter_{num_iter}", f"video_{iterator:03d}")
                    os.makedirs(save_dir_frame, exist_ok=True)
                    Image.fromarray(torch.cat([rain_display, out_display, norain_display], dim=2).permute(1, 2, 0).cpu().numpy()).save(
                        os.path.join(save_dir_frame, f"frame_{k:03d}_compare.png")
                    )

            test_bar.set_postfix(PSNR=total_psnr / count, SSIM=total_ssim / count)
            
    return total_psnr / count, total_ssim / count


if __name__ == '__main__':
    args = parse_args()
    test_dataset = VideoFrameDataset('./Dataset/Test/', data_type='test')
    test_loader = DataLoader(test_dataset, 1, False, num_workers=args.workers)

    results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': [], 'Loss': []}, 0.0, 0.0
    
    model = Restormer(channels=args.channels[0], num_heads=args.num_heads[0]).cuda()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)

    train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
    total_loss, total_num = 0.0, 0
    i = 0

    length_initial_stage = args.batch_size[0] * (args.milestone[0] if len(args.milestone) > 0 else args.num_iter)
    train_dataset = VideoFrameDataset('./Dataset/Train/', data_type='train', length=length_initial_stage)
    train_loader_iter = iter(DataLoader(train_dataset, 1, True, num_workers=args.workers))

    for n_iter in train_bar:
        if i < len(args.milestone) and n_iter == args.milestone[i] + 1:
            i += 1
            start_iter_current_stage = args.milestone[i - 1] if i > 0 else 0
            end_iter_current_stage = args.milestone[i] if i < len(args.milestone) else args.num_iter
            length_current_stage = args.batch_size[i] * (end_iter_current_stage - start_iter_current_stage)
            
            train_dataset = VideoFrameDataset('./Dataset/Train/', data_type='train', length=length_current_stage)
            train_loader_iter = iter(DataLoader(train_dataset, 1, True, num_workers=args.workers))
            print(f"--- Milestone {n_iter-1} reached. Switching to batch_size={args.batch_size[i]}, length={length_current_stage} ---")

        try:
            rain_video_batch, norain_video_batch = next(train_loader_iter)
        except StopIteration:
            print("--- End of training dataset epoch. Re-initializing. ---")
            start_iter_current_stage = args.milestone[i - 1] if i > 0 else 0
            end_iter_current_stage = args.milestone[i] if i < len(args.milestone) else args.num_iter
            length_current_stage = args.batch_size[i] * (end_iter_current_stage - start_iter_current_stage)

            train_dataset = VideoFrameDataset('./Dataset/Train/', data_type='train', length=length_current_stage)
            train_loader_iter = iter(DataLoader(train_dataset, 1, True, num_workers=args.workers))
            rain_video_batch, norain_video_batch = next(train_loader_iter)

        model.train()
        rain_video = rain_video_batch.squeeze(0)
        norain_video = norain_video_batch.squeeze(0)
        
        recurrent_frames = deque(maxlen=3)
        loss_video = 0.0

        for k in range(rain_video.shape[0]):
            rain_frame = rain_video[k].cuda()
            norain_frame = norain_video[k].cuda()

            if k == 0:
                input_tensor = torch.cat([rain_frame.unsqueeze(0), rain_frame.unsqueeze(0)], dim=0)
            elif len(recurrent_frames) < 3:
                avg_recurrent = torch.mean(torch.stack(list(recurrent_frames)), dim=0)
                input_tensor = torch.cat([rain_frame.unsqueeze(0), avg_recurrent.detach().unsqueeze(0)], dim=0)
            else:
                avg_recurrent = torch.mean(torch.stack(list(recurrent_frames)), dim=0)
                input_tensor = torch.cat([rain_frame.unsqueeze(0), avg_recurrent.detach().unsqueeze(0)], dim=0)

            out = model(input_tensor)
            recurrent_frames.append(out.detach().squeeze(0))

            output_for_loss = out.squeeze(0)
            target_for_loss = norain_frame

            WEIGHT_CHARBONNIER = 1.0
            WEIGHT_PERCEPTUAL = 1.0
            WEIGHT_SSIM_LOSS = 1.0
            WEIGHT_GRAD = 1.0

            # Convert to Y channel (output is [1, 1, H, W])
            output_y = rgb_to_y(torch.clamp(output_for_loss * 255.0, 0, 255).byte().double())
            target_y = rgb_to_y(torch.clamp(target_for_loss * 255.0, 0, 255).byte().double())

            loss_video += (
                charbonnier_loss(output_for_loss, target_for_loss) * WEIGHT_CHARBONNIER +
                perceptual_loss(output_for_loss.unsqueeze(0), target_for_loss.unsqueeze(0)) * WEIGHT_PERCEPTUAL +
                (1 - ssim(output_y, target_y)) * WEIGHT_SSIM_LOSS + # FIX: Removed .unsqueeze(0) here
                compute_loss(output_for_loss.unsqueeze(0), target_for_loss.unsqueeze(0), lambda_grad=WEIGHT_GRAD)
            )

        optimizer.zero_grad()
        loss_video.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_num += rain_video.size(0) # Accumulate by number of frames processed in this video
        total_loss += loss_video.item() * rain_video.size(0) # Accumulate by number of frames processed

        train_bar.set_description(f'Train Iter: [{n_iter}/{args.num_iter}] Loss: {total_loss / total_num:.3f}')
        lr_scheduler.step()

        if n_iter % 100 == 0:
            results['Loss'].append(f'{total_loss / total_num:.3f}')
            save_loop(model, test_loader, n_iter, args, optimizer, lr_scheduler)

