import os
from PIL import Image

def convert_jpg_to_png(folder_path):
    """Convert all JPG images in a folder to PNG format"""
    converted_count = 0
    
    # Get all JPG files in the folder
    jpg_files = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if not jpg_files:
        print(f"  No JPG files found in {os.path.basename(folder_path)}")
        return 0
    
    print(f"  Converting {len(jpg_files)} JPG files to PNG...")
    
    for jpg_file in jpg_files:
        try:
            # Open the JPG image
            jpg_path = os.path.join(folder_path, jpg_file)
            img = Image.open(jpg_path)
            
            # Convert to RGB if necessary (PNG supports RGB)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create PNG filename (same base name, different extension)
            base_name = os.path.splitext(jpg_file)[0]
            png_file = f"{base_name}.png"
            png_path = os.path.join(folder_path, png_file)
            
            # Save as PNG
            img.save(png_path, 'PNG')
            
            # Remove the original JPG file
            os.remove(jpg_path)
            
            converted_count += 1
            
        except Exception as e:
            print(f"    Error converting {jpg_file}: {str(e)}")
    
    return converted_count

def main():
    # Define base paths
    train_folder = "train"
    snow_folder = os.path.join(train_folder, "snow")
    
    # Check if snow folder exists
    if not os.path.exists(snow_folder):
        print(f"Error: Snow folder '{snow_folder}' not found!")
        return
    
    total_converted = 0
    
    print("Converting JPG to PNG in snow folders from video01 to video09...")
    
    # Process folders from video01 to video09
    for i in range(1, 10):  # 1 to 9 (video01 to video09)
        folder_name = f"video{i:02d}"  # Format as video01, video02, etc.
        folder_path = os.path.join(snow_folder, folder_name)
        
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"\nProcessing {folder_name}...")
            converted = convert_jpg_to_png(folder_path)
            total_converted += converted
            print(f"  Converted {converted} images in {folder_name}")
        else:
            print(f"Warning: Folder '{folder_name}' not found in snow directory")
    
    print(f"\n=== Conversion Complete ===")
    print(f"Total images converted: {total_converted}")
    print("All JPG images in video01-video09 have been converted to PNG format")

if __name__ == "__main__":
    main()