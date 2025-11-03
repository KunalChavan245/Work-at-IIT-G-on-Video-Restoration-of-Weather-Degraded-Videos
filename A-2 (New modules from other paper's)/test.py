import os
from collections import deque
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Restormer
from utils import parse_args, VideoFrameDataset, rgb_to_y, psnr, ssim, VGGPerceptualLoss

perceptual_loss = VGGPerceptualLoss().cuda()

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

                out = torch.clamp((out * 0.5 + 0.5).mul(255), 0, 255).byte()
                norain = torch.clamp((norain * 0.5 + 0.5).mul(255), 0, 255).byte()
                rain = torch.clamp((rain * 0.5 + 0.5).mul(255), 0, 255).byte()

                y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
                current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
                total_psnr += current_psnr.item()
                total_ssim += current_ssim.item()
                count += 1

                save_path = f'{args.save_path}/{args.data_name}/{iterator}_{k}.png'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                Image.fromarray(
                    torch.cat([rain, out, norain], dim=3).squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
                ).save(save_path)

                test_bar.set_description(
                    f'Test Iter: [{num_iter}/{args.num_iter if not args.model_file else 1}] '
                    f'PSNR: {total_psnr / count:.2f} SSIM: {total_ssim / count:.3f}'
                )
    return total_psnr / count, total_ssim / count

def save_loop(net, data_loader, num_iter):
    global best_psnr, best_ssim
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter)
    results['PSNR'].append(f'{val_psnr:.2f}')
    results['SSIM'].append(f'{val_ssim:.3f}')

    if val_psnr + val_ssim > best_psnr + best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open(f'{args.save_path}/{args.data_name}.txt', 'w') as f:
            f.write(f'Iter: {num_iter} PSNR:{best_psnr:.2f} SSIM:{best_ssim:.3f}')
        torch.save(net.state_dict(), f'{args.save_path}/{args.data_name}.pth')

if __name__ == '__main__':
    args = parse_args()
    test_dataset = VideoFrameDataset('./Dataset/Test/', data_type='test')
    test_loader = DataLoader(test_dataset, 1, False, num_workers=args.workers)

    results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': []}, 0.0, 0.0
    model = Restormer(channels=16).cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("PARAMETERS ARE::::::::::::::", total_params)

    if args.model_file:
        checkpoint = torch.load(args.model_file)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        save_loop(model, test_loader, 1)
    else:
        print("NO CHECKPOINT FOUND :( ")
