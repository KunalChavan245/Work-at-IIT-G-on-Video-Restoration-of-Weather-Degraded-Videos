import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from collections import deque
import numpy as np # For random seed
import random # For random seed

# Import your model and utils functions
# Ensure model.py and utils.py are in the same directory or accessible via PYTHONPATH
from model import Restormer, GradientBranch
from utils import VideoFrameDataset, parse_args, rgb_to_y, psnr, ssim, VGGPerceptualLoss

# Global instance for perceptual loss and gradient branch to avoid re-instantiation per trial
# Note: Ensure VGGPerceptualLoss is configured for your data's normalization range
global_perceptual_loss = VGGPerceptualLoss().cuda()
global_grad_fn_for_loss = GradientBranch().cuda()

# Define loss components (from your main.py)
def compute_loss(pred, gt, lambda_grad=0.1):
    # Removed with torch.no_grad() as per previous fix to allow gradient flow
    grad_pred = global_grad_fn_for_loss(pred)
    grad_gt = global_grad_fn_for_loss(gt)
    loss_rgb = F.l1_loss(pred, gt)
    loss_grad = F.l1_loss(grad_pred, grad_gt)
    return loss_rgb + lambda_grad * loss_grad

def charbonnier_loss(x, y, eps=1e-3):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps ** 2))

# Modified test_loop for evaluation during tuning (simplified for speed)
def evaluate_model(net, data_loader, device):
    net.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    
    # Limit evaluation to a smaller subset for faster tuning trials
    # Adjust this number based on your dataset size and desired tuning speed
    EVAL_LIMIT = 50 # Evaluate on first 50 frames of the test set

    with torch.no_grad():
        for i, (rain_video, norain_video) in enumerate(data_loader):
            if i >= EVAL_LIMIT:
                break # Limit evaluation frames

            recurrent_frames = deque(maxlen=3)
            for k, rain_frame in enumerate(rain_video):
                norain_frame = norain_video[k]
                rain, norain = rain_frame.to(device), norain_frame.to(device)

                if k == 0 or len(recurrent_frames) < 3:
                    input_tensor = torch.cat([rain, rain], dim=0)
                else:
                    avg_recurrent = torch.mean(torch.stack(list(recurrent_frames)), dim=0)
                    input_tensor = torch.cat([rain, avg_recurrent.detach()], dim=0)

                out = net(input_tensor)
                recurrent_frames.append(out.detach())

                # Convert to [0, 255] range for PSNR/SSIM calculation
                out_display = torch.clamp((out * 0.5 + 0.5).mul(255), 0, 255).byte()
                norain_display = torch.clamp((norain * 0.5 + 0.5).mul(255), 0, 255).byte()
                
                y, gt = rgb_to_y(out_display.double()), rgb_to_y(norain_display.double())
                total_psnr += psnr(y, gt).item()
                total_ssim += ssim(y, gt).item()
                count += 1
    
    if count == 0:
        return 0.0, 0.0 # Return 0 if no samples were processed

    return total_psnr / count, total_ssim / count

# Objective function for Optuna
def objective(trial: optuna.Trial):
    # Ensure reproducibility for each trial
    seed = trial.suggest_int('seed', 0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True) # Learning rate
    channels = trial.suggest_categorical("channels", [16, 24, 32]) # Base channels for the model
    num_heads = trial.suggest_categorical("num_heads", [4, 8, 16]) # Number of attention heads
    
    # Loss weights
    weight_charbonnier = trial.suggest_float("weight_charbonnier", 1.0, 10.0)
    weight_perceptual = trial.suggest_float("weight_perceptual", 1.0, 10.0)
    weight_ssim_loss = trial.suggest_float("weight_ssim_loss", 1.0, 5.0) # Renamed to avoid confusion with metric
    weight_grad = trial.suggest_float("weight_grad", 0.1, 2.0)

    # Training parameters for each trial (reduced for tuning speed)
    # Adjust these based on how long you can afford each trial to run
    NUM_TRAINING_ITERATIONS_PER_TRIAL = 200 # Run for a limited number of iterations
    BATCH_SIZE_PER_TRIAL = 1 # Keep your batch size as 1 as per your DataLoader setup

    # Initialize model
    model = Restormer(channels=channels, num_heads=num_heads).to(device)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine annealing scheduler (adjust T_max to match NUM_TRAINING_ITERATIONS_PER_TRIAL)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_TRAINING_ITERATIONS_PER_TRIAL, eta_min=1e-7)

    # DataLoaders for the tuning process (simplified for speed)
    # Using a smaller length for training dataset during tuning
    tuning_train_dataset_length = BATCH_SIZE_PER_TRIAL * NUM_TRAINING_ITERATIONS_PER_TRIAL * 2 # Roughly 2 times to ensure enough frames
    train_dataset = VideoFrameDataset('./Dataset/Train/', data_type='train', length=tuning_train_dataset_length)
    train_loader = iter(DataLoader(train_dataset, BATCH_SIZE_PER_TRIAL, shuffle=True, num_workers=0)) # num_workers=0 for debugging, increase for speed

    # Test DataLoader (same as your main script)
    test_dataset = VideoFrameDataset('./Dataset/Test/', data_type='test')
    test_loader = DataLoader(test_dataset, 1, False, num_workers=0) # num_workers=0 for tuning, increase for speed

    # Training loop for one trial
    for n_iter in range(1, NUM_TRAINING_ITERATIONS_PER_TRIAL + 1):
        model.train()
        try:
            rain_video, norain_video = next(train_loader)
        except StopIteration:
            # Re-initialize train_loader if it runs out of data
            train_dataset = VideoFrameDataset('./Dataset/Train/', data_type='train', length=tuning_train_dataset_length)
            train_loader = iter(DataLoader(train_dataset, BATCH_SIZE_PER_TRIAL, shuffle=True, num_workers=0))
            rain_video, norain_video = next(train_loader)

        recurrent_frames = deque(maxlen=3)
        loss_video_batch = 0.0 # Accumulate loss for all frames in the video sequence for the batch
        
        # Process each frame in the video sequence
        for k, rain_frame in enumerate(rain_video):
            norain_frame = norain_video[k]
            rain, norain = rain_frame.to(device), norain_frame.to(device)

            if k == 0 or len(recurrent_frames) < 3:
                input_tensor = torch.cat([rain, rain], dim=0)
            else:
                avg_recurrent = torch.mean(torch.stack(list(recurrent_frames)), dim=0)
                input_tensor = torch.cat([rain, avg_recurrent.detach()], dim=0)

            out = model(input_tensor)
            recurrent_frames.append(out.detach())

            output = out * 0.5 + 0.5
            target = norain * 0.5 + 0.5

            loss_video_batch += (
                charbonnier_loss(output, target) * weight_charbonnier +
                global_perceptual_loss(output, target) * weight_perceptual +
                (1 - ssim(output, target)) * weight_ssim_loss + # Use the suggested SSIM loss weight
                compute_loss(output, target, lambda_grad=weight_grad) * 1
            )
        
        optimizer.zero_grad()
        loss_video_batch.backward()
        
        # Optional: Gradient clipping if still experiencing exploding gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        
        optimizer.step()
        lr_scheduler.step()

        # Report intermediate value to Optuna for pruning
        if n_iter % (NUM_TRAINING_ITERATIONS_PER_TRIAL // 10) == 0: # Report every 10% of iterations
            val_psnr, val_ssim = evaluate_model(model, test_loader, device) # Get both PSNR and SSIM
            # Report a combined metric to Optuna for pruning
            # You can adjust the weights here if one metric is more important than the other
            # For example, val_psnr + val_ssim * 10 if SSIM is more critical
            trial.report(val_psnr + val_ssim, n_iter) 
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # Final evaluation after training for this trial
    final_psnr, final_ssim = evaluate_model(model, test_loader, device)
    
    # Return a combined metric to maximize (e.g., PSNR + SSIM)
    # This tells Optuna to find parameters that optimize both
    return final_psnr + final_ssim 

if __name__ == '__main__':
    # Set up Optuna study
    # Use ASHA pruner to stop unpromising trials early (more efficient)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    
    # Use a storage to save study results (e.g., SQLite database)
    # This allows you to resume tuning or analyze results later
    # Example: optuna.create_study(study_name="restormer_tuning", storage="sqlite:///restormer_tuning.db", load_if_exists=True, direction="maximize", pruner=pruner)
    # Changed direction to "maximize" as we are maximizing PSNR + SSIM
    study = optuna.create_study(direction="maximize", pruner=pruner) 

    # Run the optimization (e.g., 100 trials)
    print("Starting hyperparameter optimization...")
    study.optimize(objective, n_trials=100, timeout=7200) # Changed n_trials to 100

    print("\nHyperparameter optimization finished.")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}")

    print("\nBest trial:")
    trial = study.best_trial

    print(f"  Value (Max PSNR + SSIM): {trial.value:.4f}") # Updated print statement
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
