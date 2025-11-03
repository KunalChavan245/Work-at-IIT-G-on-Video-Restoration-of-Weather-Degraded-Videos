import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import glob

def split_image_into_three_parts(image):
    """
    Split image into three equal vertical parts
    Returns: weather_affected, output, ground_truth
    """
    height, width = image.shape[:2]
    part_width = width // 3
    
    weather_affected = image[:, 0:part_width]
    output = image[:, part_width:2*part_width]
    ground_truth = image[:, 2*part_width:3*part_width]
    
    return weather_affected, output, ground_truth

def calculate_psnr_ssim(output_img, ground_truth_img):
    """
    Calculate PSNR and SSIM between output and ground truth images
    """
    # Convert to grayscale if images are colored
    if len(output_img.shape) == 3:
        output_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(ground_truth_img, cv2.COLOR_BGR2GRAY)
    else:
        output_gray = output_img
        gt_gray = ground_truth_img
    
    # Calculate PSNR
    psnr_value = psnr(gt_gray, output_gray)
    
    # Calculate SSIM
    ssim_value = ssim(gt_gray, output_gray)
    
    return psnr_value, ssim_value



def process_images_in_folder(folder_path):
    """
    Process all images in the specified folder
    """
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = set()
    
    for extension in image_extensions:
        image_files.update(glob.glob(os.path.join(folder_path, extension)))
        image_files.update(glob.glob(os.path.join(folder_path, extension.upper())))
    
    image_files = list(image_files)
    image_files.sort()  # Sort files for consistent processing order
    
    if not image_files:
        print(f"No images found in folder: {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Lists to store all PSNR and SSIM values
    all_psnr_values = []
    all_ssim_values = []
    
    # Process each image
    for i, image_path in enumerate(image_files):
        try:
            print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
            
            # Split image into three parts
            weather_affected, output, ground_truth = split_image_into_three_parts(image)
            
            # Calculate PSNR and SSIM
            psnr_val, ssim_val = calculate_psnr_ssim(output, ground_truth)
            
            # Store values
            all_psnr_values.append(psnr_val)
            all_ssim_values.append(ssim_val)
            
            print(f"  PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Calculate and print overall averages
    if all_psnr_values:
        avg_psnr = np.mean(all_psnr_values)
        avg_ssim = np.mean(all_ssim_values)
        
        print("\n" + "="*60)
        print("OVERALL RESULTS")
        print("="*60)
        print(f"Total images processed: {len(all_psnr_values)}")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"PSNR Range: {min(all_psnr_values):.4f} - {max(all_psnr_values):.4f} dB")
        print(f"SSIM Range: {min(all_ssim_values):.4f} - {max(all_ssim_values):.4f}")
        
        # Save results to file
        with open(os.path.join(folder_path, 'psnr_ssim_results.txt'), 'w') as f:
            f.write("PSNR and SSIM Results\n")
            f.write("="*30 + "\n\n")
            f.write(f"Total images processed: {len(all_psnr_values)}\n")
            f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"PSNR Range: {min(all_psnr_values):.4f} - {max(all_psnr_values):.4f} dB\n")
            f.write(f"SSIM Range: {min(all_ssim_values):.4f} - {max(all_ssim_values):.4f}\n")
        
        print(f"\nResults saved to: {os.path.join(folder_path, 'psnr_ssim_results.txt')}")
    
    else:
        print("No images were successfully processed.")

def main():
    # Specify the folder path
    folder_path = r"D:\IITG project\result\Thick"
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist!")
        print("Please make sure the folder path is correct.")
        return
    
    print(f"Starting PSNR and SSIM calculation for images in: {folder_path}")
    print("-" * 60)
    
    # Process all images in the folder
    process_images_in_folder(folder_path)

if __name__ == "__main__":
    main()



# import os
# import cv2
# import glob

# def split_image_into_three_parts(image):
#     """
#     Split image into three equal vertical parts
#     Returns: weather_affected, output, ground_truth
#     """
#     height, width = image.shape[:2]
#     part_width = width // 3
    
#     weather_affected = image[:, 0:part_width]
#     output = image[:, part_width:2*part_width]
#     ground_truth = image[:, 2*part_width:3*part_width]
    
#     return weather_affected, output, ground_truth

# def create_output_folders(base_folder):
#     """
#     Create output and ground_truth folders inside the base folder
#     """
#     output_folder = os.path.join(base_folder, "output")
#     ground_truth_folder = os.path.join(base_folder, "ground_truth")
    
#     # Create folders if they don't exist
#     os.makedirs(output_folder, exist_ok=True)
#     os.makedirs(ground_truth_folder, exist_ok=True)
    
#     return output_folder, ground_truth_folder

# def separate_and_save_images(input_folder, output_base_folder):
#     """
#     Process all images in input folder and save separated parts
#     """
#     # Get all image files
#     image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
#     image_files = set()
    
#     for extension in image_extensions:
#         image_files.update(glob.glob(os.path.join(input_folder, extension)))
#         image_files.update(glob.glob(os.path.join(input_folder, extension.upper())))
    
#     image_files = list(image_files)
#     image_files.sort()  # Sort files for consistent processing order
    
#     if not image_files:
#         print(f"No images found in folder: {input_folder}")
#         return
    
#     print(f"Found {len(image_files)} images to process")
    
#     # Create output folders
#     output_folder, ground_truth_folder = create_output_folders(output_base_folder)
    
#     print(f"Output folder: {output_folder}")
#     print(f"Ground truth folder: {ground_truth_folder}")
#     print("-" * 60)
    
#     # Process each image
#     successful_count = 0
    
#     for i, image_path in enumerate(image_files):
#         try:
#             filename = os.path.basename(image_path)
#             name_without_ext = os.path.splitext(filename)[0]
#             ext = os.path.splitext(filename)[1]
            
#             print(f"Processing image {i+1}/{len(image_files)}: {filename}")
            
#             # Read the image
#             image = cv2.imread(image_path)
#             if image is None:
#                 print(f"  Error: Could not read image: {image_path}")
#                 continue
            
#             # Get image dimensions
#             height, width = image.shape[:2]
#             print(f"  Original dimensions: {width} x {height}")
            
#             # Split image into three parts
#             weather_affected, output, ground_truth = split_image_into_three_parts(image)
            
#             # Get dimensions of separated parts
#             part_height, part_width = output.shape[:2]
#             print(f"  Each part dimensions: {part_width} x {part_height}")
            
#             # Save output image
#             output_filename = f"{name_without_ext}_output{ext}"
#             output_path = os.path.join(output_folder, output_filename)
#             cv2.imwrite(output_path, output)
            
#             # Save ground truth image
#             gt_filename = f"{name_without_ext}_gt{ext}"
#             gt_path = os.path.join(ground_truth_folder, gt_filename)
#             cv2.imwrite(gt_path, ground_truth)
            
#             print(f"  ✓ Saved: {output_filename} and {gt_filename}")
#             successful_count += 1
            
#         except Exception as e:
#             print(f"  Error processing {filename}: {str(e)}")
#             continue
    
#     print("\n" + "="*60)
#     print("SEPARATION COMPLETE")
#     print("="*60)
#     print(f"Total images processed successfully: {successful_count}/{len(image_files)}")
#     print(f"Output images saved to: {output_folder}")
#     print(f"Ground truth images saved to: {ground_truth_folder}")
    
#     # Create a summary file
#     summary_file = os.path.join(output_base_folder, "separation_summary.txt")
#     with open(summary_file, 'w') as f:
#         f.write("Image Separation Summary\n")
#         f.write("="*30 + "\n\n")
#         f.write(f"Input folder: {input_folder}\n")
#         f.write(f"Output folder: {output_folder}\n")
#         f.write(f"Ground truth folder: {ground_truth_folder}\n\n")
#         f.write(f"Total images found: {len(image_files)}\n")
#         f.write(f"Successfully processed: {successful_count}\n")
#         f.write(f"Failed: {len(image_files) - successful_count}\n\n")
#         f.write("Process: Each input image was split into 3 equal vertical parts:\n")
#         f.write("- Left part: Weather affected (not saved)\n")
#         f.write("- Middle part: Output (saved with '_output' suffix)\n")
#         f.write("- Right part: Ground truth (saved with '_gt' suffix)\n")
    
#     print(f"Summary saved to: {summary_file}")

# def main():
#     # Specify the input folder path (where your original images are)
#     input_folder = r"D:\IITG project\result\Thick"
    
#     # Specify the output base folder (where separated images will be saved)
#     output_base_folder = r"D:\IITG project\result\Separated_Images"
    
#     # Alternative path specifications (uncomment one if needed):
#     # input_folder = "D:/IITG project/result/Thick"
#     # output_base_folder = "D:/IITG project/result/Separated_Images"
    
#     # Check if input folder exists
#     if not os.path.exists(input_folder):
#         print(f"Input folder '{input_folder}' does not exist!")
#         print("Please make sure the folder path is correct.")
#         return
    
#     # Create output base folder if it doesn't exist
#     os.makedirs(output_base_folder, exist_ok=True)
    
#     print(f"Starting image separation...")
#     print(f"Input folder: {input_folder}")
#     print(f"Output base folder: {output_base_folder}")
#     print("-" * 60)
    
#     # Process all images
#     separate_and_save_images(input_folder, output_base_folder)

# if __name__ == "__main__":
#     main()