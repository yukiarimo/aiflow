import os
import argparse
from PIL import Image
import glob

def convert_and_rename_images(input_dir, start_number=1, output_dir=None):
    """
    Convert all images in a directory to PNG format and rename them sequentially.

    Args:
        input_dir (str): Directory containing images to process
        start_number (int): Starting number for sequential renaming
        output_dir (str): Optional output directory for converted images
    """
    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define target directory for saving
    target_dir = output_dir if output_dir else input_dir

    # Find all image files in the input directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    print(f"Found {len(image_files)} image files to process")

    # Process each image
    count = start_number
    for img_path in sorted(image_files):
        try:
            # Skip if it's already a PNG with the correct name format
            filename = os.path.basename(img_path)
            if filename == f"{count}.png":
                count += 1
                continue

            # Open and convert the image
            with Image.open(img_path) as img:
                # Create new filename
                new_filename = f"{count}.png"
                new_path = os.path.join(target_dir, new_filename)

                # Save as PNG
                img.save(new_path, "PNG")
                print(f"Converted and renamed: {img_path} → {new_path}")

                # Delete original if it's different and in the same directory
                if output_dir is None and os.path.abspath(img_path) != os.path.abspath(new_path):
                    os.remove(img_path)
                    print(f"Deleted original: {img_path}")

                count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Conversion complete. Processed {count - start_number} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert images to PNG and rename sequentially')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, help='Optional output directory (if not specified, input_dir is used)')
    parser.add_argument('--start', type=int, default=1, help='Starting number for sequential naming (default: 1)')

    args = parser.parse_args()

    convert_and_rename_images(args.input_dir, args.start, args.output_dir)

"""
> Example usage:

# Basic usage - convert all images in a directory and replace them
python convert_images.py --input_dir /path/to/images

# Save converted images to a different directory
python convert_images.py --input_dir /path/to/images --output_dir /path/to/output

# Start numbering from a specific number
python convert_images.py --input_dir /path/to/images --start 100

"""