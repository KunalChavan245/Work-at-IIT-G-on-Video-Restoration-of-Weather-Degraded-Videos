import os
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Restormer
from utils import parse_args, VideoFrameDataset,RainDataset, rgb_to_y, psnr, ssim, VGGPerceptualLoss


perceptual_loss = VGGPerceptualLoss().cuda()

def test_loop(net, data_loader, num_iter):
    net.eval()
    iterator = 0
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    data_loader =  iter(data_loader)
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)

        for rain_video, norain_video in test_bar:
            
            iterator +=1
            recurrent_frames = []
            for k, rain_frame in tqdm(enumerate(rain_video), desc = "Frames Processed"):
                norain_frame = norain_video[k]
                rain, norain= rain_frame.cuda(), norain_frame.cuda()
                if k==0:
                    out = torch.clamp((model(torch.cat([rain,rain]))*0.5+0.5).mul(255), 0, 255).byte()
                    recurrent_frames.append(out)
                else:
                    out = torch.clamp((model(torch.cat([rain,recurrent_frames[-1]]))*0.5+0.5).mul(255), 0, 255).byte()
                    
                    recurrent_frames.append(out)
                
                norain = torch.clamp((norain*0.5+0.5).mul(255), 0 ,255).byte()
                # computer the metrics with Y channel and double precision
                y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
                current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
                total_psnr += current_psnr.item()
                total_ssim += current_ssim.item()
                count += 1
                save_path = f'{args.save_path}/{args.data_name}/{iterator}_{k}.png'
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save(save_path)
                test_bar.set_description('Test Iter: [{}/{}] PSNR: {:.2f} SSIM: {:.3f}'
                                         .format(num_iter, 1 if args.model_file else args.num_iter,
                                                 total_psnr / count, total_ssim / count))
    return total_psnr / count, total_ssim / count


def save_loop(net, data_loader, num_iter):
    global best_psnr, best_ssim
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter)
    results['PSNR'].append('{:.2f}'.format(val_psnr))
    results['SSIM'].append('{:.3f}'.format(val_ssim))
    # save statistics
    # data_frame = pd.DataFrame(data=results, index=range(1, (num_iter if args.model_file else num_iter // 1000) + 1))
    # data_frame.to_csv('{}/{}.csv'.format(args.save_path, args.data_name), index_label='Iter', float_format='%.3f')
    if val_psnr > best_psnr and val_ssim > best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open('{}/{}.txt'.format(args.save_path, args.data_name), 'w') as f:
            f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr, best_ssim))
        torch.save(model.state_dict(), '{}/{}.pth'.format(args.save_path, args.data_name))

# from ptflops import get_model_complexity_info
        
if __name__ == '__main__':
    args = parse_args()
    test_dataset = VideoFrameDataset('/home/cvpr_int_5/VideoResto/Dataset_resized/Test', data_type = 'test')
    test_loader = DataLoader(test_dataset, 1, shuffle = False, num_workers=args.workers)

    results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': []}, 0.0, 0.0
    model  = Restormer(channels = 16).cuda()
    # macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
                
    # exit(0)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("PARAMETERS ARE::::::::::::::",pytorch_total_params)
    # exit(0)   
    
    

    
    if args.model_file:
        if args.finetune is True:
            print('loaded existing model....')
            model.load_state_dict(torch.load(args.model_file))
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)
            total_loss, total_num, loss_video, results['Loss'], i = 0.0, 0, 0.0, [], 0
            train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
            for n_iter in train_bar:
                    # progressive learning
                if n_iter == 1 or n_iter - 1 in args.milestone:
                    end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
                    start_iter = args.milestone[i - 1] if i > 0 else 0
                    length = args.batch_size[i] * (end_iter - start_iter)

                    train_loader = VideoFrameDataset('/home/cvpr_int_5/VideoResto/Dataset_resized/Train', data_type = 'train', length = length)
                    train_loader = iter(DataLoader(train_loader, 1, True, num_workers=args.workers))
                    i += 1
                # train
                
                rain_video, norain_video = next(train_loader)
                recurrent_frames = []
                for k, rain_frame in enumerate(rain_video):
                    loss_video = 0.0
                    model.train()

                    norain_frame = norain_video[k]
                    rain, norain= rain_frame.cuda(), norain_frame.cuda()
                    if k==0:

                        out = model(torch.cat([rain, rain]))
                        output = out
                        recurrent_frames.append(output.detach())
                                    
                    else:
                        # print(torch.cat([rain, recurrent_frames[-1]]).shape)
                        out = model(torch.cat([rain, recurrent_frames[-1]]))
                        output = out
                        # print("Rain Frame MIN MAX ", rain.min(), rain.max())
                        # print("OUTPUT MIN MAX ", out.min(), out.max())

                        recurrent_frames.pop()
                        recurrent_frames.append(output.detach())
                    loss_video += F.l1_loss(out*0.5+0.5, norain*0.5+0.5)*10.0 + perceptual_loss(out*0.5+0.5, norain*0.5+0.5) * 10



                optimizer.zero_grad()
                loss_video.backward()
                optimizer.step()
                total_num += rain.size(0)
                total_loss += loss_video.item() * rain.size(0)
                train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
                                          .format(n_iter, args.num_iter, total_loss / total_num))

                lr_scheduler.step()
                if n_iter %500 == 0:
                    results['Loss'].append('{:.3f}'.format(total_loss / total_num))
                    save_loop(model, test_loader, n_iter)
        else:
            model.load_state_dict(torch.load(args.model_file))
            save_loop(model, test_loader, 1)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)
        total_loss, total_num, results['Loss'], i = 0.0, 0, [], 0
        train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
        for n_iter in train_bar:
                # progressive learning
            if n_iter == 1 or n_iter - 1 in args.milestone:
                end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
                start_iter = args.milestone[i - 1] if i > 0 else 0
                length = args.batch_size[i] * (end_iter - start_iter)

                train_loader = VideoFrameDataset('/home/cvpr_int_5/VideoResto/Dataset_resized/Train', data_type = 'train', length = length)
                train_loader = iter(DataLoader(train_loader, 1, True, num_workers=args.wogrkers))
                i += 1
            # train
            
            rain_video, norain_video = next(train_loader)
            recurrent_frames = []
            for k, rain_frame in enumerate(rain_video):
                loss_video = 0.0
                model.train()

                norain_frame = norain_video[k]
                rain, norain= rain_frame.cuda(), norain_frame.cuda()
                if k==0:

                    out = model(torch.cat([rain, rain]))
                    output = out
                    recurrent_frames.append(output.detach())
                                
                else:
                    # print(torch.cat([rain, recurrent_frames[-1]]).shape)
                    out = model(torch.cat([rain, recurrent_frames[-1].detach()]))
                    output = out
                    # print("Rain Frame MIN MAX ", rain.min(), rain.max())
                    # print("OUTPUT MIN MAX ", out.min(), out.max())

                    recurrent_frames.pop()
                    recurrent_frames.append(output.detach())
                loss_video += F.l1_loss(out*0.5+0.5, norain*0.5+0.5) + perceptual_loss(out*0.5+0.5, norain*0.5+0.5)



            optimizer.zero_grad()
            loss_video.backward()
            optimizer.step()
            total_num += rain.size(0)
            total_loss += loss_video.item() * rain.size(0)
            train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
                                      .format(n_iter, args.num_iter, total_loss / total_num))

            lr_scheduler.step()
            if n_iter %100 == 0:
                results['Loss'].append('{:.3f}'.format(total_loss / total_num))
                save_loop(model, test_loader, n_iter)
