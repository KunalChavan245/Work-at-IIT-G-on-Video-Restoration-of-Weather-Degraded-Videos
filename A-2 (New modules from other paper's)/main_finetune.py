# === main_finetune.py ===
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

from model import Restormer
from utils import parse_args, VideoFrameDataset, rgb_to_y, psnr, ssim, VGGPerceptualLoss

perceptual_loss = VGGPerceptualLoss().cuda()

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

                out = model(input_tensor)
                recurrent_frames.append(out.detach())
                out = torch.clamp((out * 0.5 + 0.5).mul(255), 0, 255).byte()
                norain = torch.clamp((norain * 0.5 + 0.5).mul(255), 0, 255).byte()
                y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
                total_psnr += psnr(y, gt).item()
                total_ssim += ssim(y, gt).item()
                count += 1
                save_path = f'{args.save_path}/{args.data_name}/{iterator}_{k}.png'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).cpu().numpy()).save(save_path)
                test_bar.set_description(f'Test Iter: [{num_iter}/{args.num_iter}] PSNR: {total_psnr / count:.2f} SSIM: {total_ssim / count:.3f}')
    return total_psnr / count, total_ssim / count

def save_loop(net, data_loader, num_iter):
    global best_psnr, best_ssim
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter)
    results['PSNR'].append(f'{val_psnr:.2f}')
    results['SSIM'].append(f'{val_ssim:.3f}')
    if val_psnr > best_psnr and val_ssim > best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open(f'{args.save_path}/{args.data_name}.txt', 'w') as f:
            f.write(f'Iter: {num_iter} PSNR:{best_psnr:.2f} SSIM:{best_ssim:.3f}')
        torch.save(model.state_dict(), f'{args.save_path}/{args.data_name}.pth')

if __name__ == '__main__':
    args = parse_args()
    test_dataset = VideoFrameDataset('./Dataset/Test/', data_type='test')
    test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=args.workers)

    results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': [], 'Loss': []}, 0.0, 0.0
    model = Restormer(channels=16).cuda()

    if args.model_file and args.finetune:
        print("Loaded existing model...")
        model.load_state_dict(torch.load(args.model_file))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)

    i = 0
    train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
    total_loss, total_num = 0.0, 0
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

            loss_video += (
                charbonnier_loss(output, target) * 5 +
                perceptual_loss(output, target) * 5 +
                (1 - ssim(output, target)) * 2
            )

        optimizer.zero_grad()
        loss_video.backward()
        optimizer.step()
        total_num += rain.size(0)
        total_loss += loss_video.item() * rain.size(0)

        train_bar.set_description(f'Train Iter: [{n_iter}/{args.num_iter}] Loss: {total_loss / total_num:.3f}')
        lr_scheduler.step()

        if n_iter % 500 == 0:
            results['Loss'].append(f'{total_loss / total_num:.3f}')
            save_loop(model, test_loader, n_iter)

