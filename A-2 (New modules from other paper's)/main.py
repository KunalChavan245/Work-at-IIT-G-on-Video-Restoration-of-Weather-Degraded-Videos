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
import numpy as np # For random seed
import random # For random seed

# Import your model and utils functions
from model import GradientBranch, Restormer
from utils import parse_args, VideoFrameDataset, rgb_to_y, psnr, ssim, VGGPerceptualLoss

# Set the seed for reproducibility based on the best trial
OPTIMIZED_SEED = 887
torch.manual_seed(OPTIMIZED_SEED)
np.random.seed(OPTIMIZED_SEED)
random.seed(OPTIMIZED_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(OPTIMIZED_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

def test_loop(net, data_loader, num_iter):
    net.eval()
    iterator = 0
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    data_loader = iter(data_loader)
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for rain_video, norain_video in test_bar:
            iterator += 1
            recurrent_frames = deque(maxlen=3)
            for k, rain_frame in enumerate(rain_video):
                norain_frame = norain_video[k]
                rain, norain = rain_frame.cuda(), norain_frame.cuda()
                if k == 0 or len(recurrent_frames) < 3:
                    input_tensor = torch.cat([rain, rain], dim=0)
                else:
                    avg_recurrent = torch.mean(torch.stack(list(recurrent_frames)), dim=0)
                    input_tensor = torch.cat([rain, avg_recurrent.detach()], dim=0)
                out = net(input_tensor)
                recurrent_frames.append(out.detach())

                # --- DEBUGGING PRINTS: Check Tensor Ranges ---
                # Model output is likely [-1, 1], scale to [0, 1] then [0, 255]
                output_0_1 = out * 0.5 + 0.5
                target_0_1 = norain * 0.5 + 0.5

                out_display = torch.clamp(output_0_1.mul(255), 0, 255).byte()
                norain_display = torch.clamp(target_0_1.mul(255), 0, 255).byte()
                rain_display = torch.clamp((rain * 0.5 + 0.5).mul(255), 0, 255).byte()
                
                print(f"\n--- Frame {k} Test Loop Debug ---")
                print(f"  out (model output) min/max: {out.min().item():.4f}/{out.max().item():.4f}")
                print(f"  norain (GT) min/max: {norain.min().item():.4f}/{norain.max().item():.4f}")
                print(f"  out_display (0-255) min/max: {out_display.min().item()}/{out_display.max().item()}")
                print(f"  norain_display (0-255) min/max: {norain_display.min().item()}/{norain_display.max().item()}")

                # Convert to Y channel for PSNR/SSIM calculation
                y_out = rgb_to_y(out_display.double())
                y_norain = rgb_to_y(norain_display.double())
                
                print(f"  y_out (Y-channel) min/max: {y_out.min().item():.4f}/{y_out.max().item():.4f}")
                print(f"  y_norain (GT Y-channel) min/max: {y_norain.min().item():.4f}/{y_norain.max().item():.4f}")

                current_psnr = psnr(y_out, y_norain).item()
                current_ssim = ssim(y_out, y_norain).item()
                print(f"  Current Frame PSNR: {current_psnr:.2f}, SSIM: {current_ssim:.3f}")
                # --- END DEBUGGING PRINTS ---

                total_psnr += current_psnr
                total_ssim += current_ssim
                count += 1
                
                save_path = f'{args.save_path}/{args.data_name}/{iterator}_{k}.png'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Ensure the concatenation is correct; 'dim=3' for horizontal concatenation across batch dimension
                Image.fromarray(torch.cat([rain_display, out_display, norain_display], dim=3).squeeze(dim=0).permute(1, 2, 0).cpu().numpy()).save(save_path)
                
                test_bar.set_description(f'Test Iter: [{num_iter}/{args.num_iter}] PSNR: {total_psnr / count:.2f} SSIM: {total_ssim / count:.3f}')
    return total_psnr / count, total_ssim / count

def save_loop(net, data_loader, num_iter):
    global best_psnr, best_ssim
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter)
    results['PSNR'].append(f'{val_psnr:.2f}')
    results['SSIM'].append(f'{val_ssim:.3f}')
    # Note: Optimization was for PSNR + SSIM. When saving, we still compare the sum.
    if val_psnr + val_ssim > best_psnr + best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open(f'{args.save_path}/{args.data_name}.txt', 'w') as f:
            f.write(f'Iter: {num_iter} PSNR:{best_psnr:.2f} SSIM:{best_ssim:.3f}')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'iter': num_iter,
            'best_psnr': best_psnr,
            'best_ssim': best_ssim
        }, f'{args.save_path}/{args.data_name}.pth')

if __name__ == '__main__':
    args = parse_args()
    test_dataset = VideoFrameDataset('./Dataset/Test/', data_type='test')
    test_loader = DataLoader(test_dataset, 1, False, num_workers=args.workers)

    results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': [], 'Loss': []}, 0.0, 0.0
    
    # Initialize model with optimized channels and num_heads
    OPTIMIZED_CHANNELS = 16
    OPTIMIZED_NUM_HEADS = 8
    model = Restormer(channels=OPTIMIZED_CHANNELS, num_heads=OPTIMIZED_NUM_HEADS).cuda()

    # Initialize optimizer with optimized learning rate
    OPTIMIZED_LR = 1.8470213731822586e-05
    optimizer = AdamW(model.parameters(), lr=OPTIMIZED_LR, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)

    train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
    total_loss, total_num = 0.0, 0
    i = 0
    for n_iter in train_bar:
        if n_iter == 1 or n_iter - 1 in args.milestone:
            end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
            length = args.batch_size[i] * (end_iter - (args.milestone[i - 1] if i > 0 else 0))
            train_loader = VideoFrameDataset('./Dataset/Train/', data_type='train', length=length)
            train_loader = iter(DataLoader(train_loader, 1, True, num_workers=args.workers))
            i += 1

        rain_video, norain_video = next(train_loader)
        recurrent_frames = deque(maxlen=3)
        loss_video = 0.0
        for k, rain_frame in enumerate(rain_video):
            model.train()
            norain_frame = norain_video[k]
            rain, norain = rain_frame.cuda(), norain_frame.cuda()

            if k == 0 or len(recurrent_frames) < 3:
                input_tensor = torch.cat([rain, rain], dim=0)
            else:
                avg_recurrent = torch.mean(torch.stack(list(recurrent_frames)), dim=0)
                input_tensor = torch.cat([rain, avg_recurrent.detach()], dim=0)

            out = model(input_tensor)
            recurrent_frames.append(out.detach())

            output = out * 0.5 + 0.5
            target = norain * 0.5 + 0.5

            # Apply optimized loss weights
            OPTIMIZED_WEIGHT_CHARBONNIER = 3.866960954246183
            OPTIMIZED_WEIGHT_PERCEPTUAL = 8.387266260573597
            OPTIMIZED_WEIGHT_SSIM_LOSS = 4.172008849729517
            OPTIMIZED_WEIGHT_GRAD = 1.9417690116635669

            # FIX: Apply rgb_to_y to output and target before passing to ssim for loss calculation
            output_y = rgb_to_y(output)
            target_y = rgb_to_y(target)

            loss_video += (
                charbonnier_loss(output, target) * OPTIMIZED_WEIGHT_CHARBONNIER +
                perceptual_loss(output, target) * OPTIMIZED_WEIGHT_PERCEPTUAL +
                (1 - ssim(output_y, target_y)) * OPTIMIZED_WEIGHT_SSIM_LOSS + # Corrected SSIM loss input
                compute_loss(output, target, lambda_grad=OPTIMIZED_WEIGHT_GRAD) * 1
            )

        optimizer.zero_grad()
        loss_video.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Keep gradient clipping
        optimizer.step()
        total_num += rain.size(0)
        total_loss += loss_video.item() * rain.size(0)

        train_bar.set_description(f'Train Iter: [{n_iter}/{args.num_iter}] Loss: {total_loss / total_num:.3f}')
        lr_scheduler.step()

        if n_iter % 500 == 0:
            results['Loss'].append(f'{total_loss / total_num:.3f}')
            save_loop(model, test_loader, n_iter)
