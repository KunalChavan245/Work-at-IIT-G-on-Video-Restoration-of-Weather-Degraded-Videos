





# import os

# def rename_folders_numerically(base_folder, start_index=0):
#     """
#     Rename folders numerically starting from a given index
#     Returns the next available index
#     """
#     # List all subdirectories
#     folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
#     folders.sort()  # Sort alphabetically before renaming for consistency

#     current_index = start_index
    
#     for folder in folders:
#         old_path = os.path.join(base_folder, folder)
#         new_name = f"{current_index:05d}"  # Format: 00000, 00001, etc.
#         new_path = os.path.join(base_folder, new_name)

#         if old_path != new_path:
#             os.rename(old_path, new_path)
#             print(f"Renamed: {folder} → {new_name}")
        
#         current_index += 1
    
#     return current_index

# def main():
#     base_train = "Train"
    
#     # Define the dataset folders in order
#     datasets = ["KITTI", "Rain", "Revide"]
    
#     # Check if base train folder exists
#     if not os.path.exists(base_train):
#         print(f"Error: '{base_train}' folder not found!")
#         return
    
#     current_index = 1
    
#     for dataset_name in datasets:
#         dataset_path = os.path.join(base_train, dataset_name)
#         train_path = os.path.join(dataset_path, "train")
        
#         # Check if dataset folder exists
#         if not os.path.exists(dataset_path):
#             print(f"Warning: '{dataset_name}' folder not found in '{base_train}'. Skipping...")
#             continue
            
#         # Check if train folder exists inside dataset folder
#         if not os.path.exists(train_path):
#             print(f"Warning: 'train' folder not found inside '{dataset_name}'. Skipping...")
#             continue
        
#         print(f"\nProcessing {dataset_name}...")
#         print(f"Renaming folders inside '{train_path}' starting from index {current_index:05d}...")
        
#         # Rename folders and get the next available index
#         current_index = rename_folders_numerically(train_path, current_index)
        
#         print(f"Completed {dataset_name}. Next index will be: {current_index:05d}")
    
#     print(f"\n✅ All dataset folder renaming complete!")
#     print(f"Total folders processed: {current_index}")

# if __name__ == "__main__":
#     main()
















# # import os
# # import re

# # def rename_folders_in_dataset(base_path, dataset_name, prefix):
# #     """
# #     Rename folders in dataset from 00000 format to prefix_01 format
    
# #     Args:
# #         base_path: Base path (e.g., "train")
# #         dataset_name: Dataset folder name (e.g., "KITTI", "Rain", "Revide")
# #         prefix: New folder prefix (e.g., "snow", "rain", "haze")
# #     """
# #     dataset_train_path = os.path.join(base_path, dataset_name, "train")
    
# #     # Check if dataset/train path exists
# #     if not os.path.exists(dataset_train_path):
# #         print(f"Warning: Path '{dataset_train_path}' not found! Skipping {dataset_name}...")
# #         return 0
    
# #     # Get all subdirectories in dataset/train
# #     folders = [f for f in os.listdir(dataset_train_path) if os.path.isdir(os.path.join(dataset_train_path, f))]
# #     folders.sort()  # Sort to maintain order
    
# #     total_folders_renamed = 0
    
# #     print(f"\n=== Processing {dataset_name} Dataset ===")
    
# #     for folder in folders:
# #         # Check if folder follows the 00000 pattern (5 digits)
# #         match = re.match(r'^(\d{5})$', folder)
        
# #         if match:
# #             number_part = match.group(1)  # The 5-digit number (e.g., "00025")
            
# #             # Convert to integer and then to prefix_XX format
# #             folder_number = int(number_part)
# #             new_name = f"{prefix}_{folder_number:02d}"  # Keep the original number
            
# #             old_path = os.path.join(dataset_train_path, folder)
# #             new_path = os.path.join(dataset_train_path, new_name)
            
# #             if old_path != new_path:
# #                 try:
# #                     os.rename(old_path, new_path)
# #                     print(f"  Renamed: {folder} → {new_name}")
# #                     total_folders_renamed += 1
# #                 except OSError as e:
# #                     print(f"  Error renaming {folder}: {e}")
# #         else:
# #             print(f"  Skipped: {folder} (doesn't match 00000 pattern)")
    
# #     print(f"\n✅ {dataset_name} folder renaming complete!")
# #     print(f"Folders renamed in {dataset_name}: {total_folders_renamed}")
    
# #     return total_folders_renamed

# # def main():
# #     base_train = "Train"
    
# #     # Check if base train folder exists
# #     if not os.path.exists(base_train):
# #         print(f"Error: '{base_train}' folder not found!")
# #         return
    
# #     # Define datasets and their corresponding prefixes
# #     datasets_config = [
# #         ("KITTI", "snow"),
# #         ("Rain", "rain"),
# #         ("Revide", "haze")
# #     ]
    
# #     print("Starting dataset folder renaming...")
# #     print("Converting folder names from 00000 format to prefix_XX format")
    
# #     total_folders_processed = 0
    
# #     for dataset_name, prefix in datasets_config:
# #         folders_renamed = rename_folders_in_dataset(base_train, dataset_name, prefix)
# #         total_folders_processed += folders_renamed
    
# #     print(f"\n🎉 ALL DATASET FOLDER RENAMING COMPLETE! 🎉")
# #     print(f"Total folders renamed across all datasets: {total_folders_processed}")
# #     print("\nSummary:")
# #     print("- KITTI folders: 00001 → snow_01, 00002 → snow_02, etc.")
# #     print("- Rain folders:  00025 → rain_25, 00026 → rain_26, etc.")
# #     print("- Revide folders: 00037 → haze_37, 00038 → haze_38, etc.")

# # if __name__ == "__main__":
# #     main()

# Check CUDA availability and compatibility
import torch
import subprocess
import sys

print("=== CUDA Compatibility Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available in PyTorch: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available in current PyTorch installation")

# Check if NVIDIA GPU is present
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("\n=== NVIDIA GPU Detected ===")
        print("nvidia-smi output:")
        print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
    else:
        print("\n=== No NVIDIA GPU or nvidia-smi not found ===")
except FileNotFoundError:
    print("\n=== nvidia-smi not found - No NVIDIA GPU driver installed ===")

print(f"\nPython executable: {sys.executable}")
print("To install CUDA-enabled PyTorch, visit: https://pytorch.org/get-started/locally/")