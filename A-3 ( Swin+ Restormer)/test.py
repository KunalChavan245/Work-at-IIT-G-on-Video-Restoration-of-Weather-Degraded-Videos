import os
from collections import deque
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Restormer # Import the updated Restormer model
from utils import parse_args, VideoFrameDataset, rgb_to_y, psnr, ssim, VGGPerceptualLoss

perceptual_loss = VGGPerceptualLoss().cuda() # Ensure it's on CUDA if available

def test_loop(net, data_loader, num_iter, args=None): # Added args parameter
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
                
                y_out = rgb_to_y(out_display.double())
                y_norain = rgb_to_y(norain_display.double())
                
                current_psnr = psnr(y_out, y_norain).item()
                current_ssim = ssim(y_out, y_norain).item()

                total_psnr += current_psnr
                total_ssim += current_ssim
                count += 1
                
                if args and args.save_path:
                    save_dir_frame = os.path.join(args.save_path, args.data_name, f"test_results_iter_{num_iter}", f"video_{iterator:03d}")
                    os.makedirs(save_dir_frame, exist_ok=True)
                    Image.fromarray(torch.cat([rain_display, out_display, norain_display], dim=2).permute(1, 2, 0).cpu().numpy()).save(
                        os.path.join(save_dir_frame, f"frame_{k:03d}_compare.png")
                    )

            test_bar.set_postfix(PSNR=total_psnr / count, SSIM=total_ssim / count)
            
    return total_psnr / count, total_ssim / count

def save_loop(net, data_loader, num_iter, args):
    global best_psnr, best_ssim, results
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter, args)
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

    # FIX: Initialize 'results' with 'Loss' key as well, consistent with main.py
    results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': [], 'Loss': []}, 0.0, 0.0
    
    # Initialize model with channels from args
    model = Restormer(channels=args.channels[0], num_heads=args.num_heads[0]).cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", total_params)

    if args.model_file:
        checkpoint = torch.load(args.model_file)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {args.model_file} at iteration {checkpoint.get('iter', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model from {args.model_file}")
        
        test_loop(model, test_loader, 1, args)
    else:
        print("No checkpoint file provided for testing. Use --model_file argument to load a pre-trained model.")

