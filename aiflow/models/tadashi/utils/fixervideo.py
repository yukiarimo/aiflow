#!/usr/bin/env python3
# filepath: /Users/yuki/Documents/tadashi/fixervideo.py
import os
import argparse
import glob
import subprocess
import shutil

def convert_and_rename_videos(input_dir, start_number=1, output_dir=None):
    """
    Convert all videos in a directory to MP4 format and rename them sequentially.

    Args:
        input_dir (str): Directory containing videos to process
        start_number (int): Starting number for sequential renaming
        output_dir (str): Optional output directory for converted videos
    """
    # Check if ffmpeg is installed
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg is not installed. Please install it to convert videos.")
        return

    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define target directory for saving
    target_dir = output_dir if output_dir else input_dir

    # Find all video files in the input directory
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v', '*.3gp']
    video_files = []

    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
        video_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    print(f"Found {len(video_files)} video files to process")

    # Process each video
    count = start_number
    for video_path in sorted(video_files):
        try:
            # Skip if it's already an MP4 with the correct name format
            filename = os.path.basename(video_path)
            if filename == f"{count}.mp4":
                count += 1
                continue

            # Create new filename
            new_filename = f"{count}.mp4"
            new_path = os.path.join(target_dir, new_filename)

            # Convert to MP4 using ffmpeg
            print(f"Converting: {video_path} → {new_path}")

            # Use ffmpeg to convert the video
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-c:v', 'libx264',  # H.264 video codec
                '-c:a', 'aac',      # AAC audio codec
                '-strict', 'experimental',
                '-b:a', '192k',
                '-y',               # Overwrite output files without asking
                new_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            print(f"Converted and renamed: {video_path} → {new_path}")

            # Delete original if it's different and in the same directory
            if output_dir is None and os.path.abspath(video_path) != os.path.abspath(new_path):
                os.remove(video_path)
                print(f"Deleted original: {video_path}")

            count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error converting {video_path}: {e}")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    print(f"Conversion complete. Processed {count - start_number} videos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert videos to MP4 and rename sequentially')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, help='Optional output directory (if not specified, input_dir is used)')
    parser.add_argument('--start', type=int, default=1, help='Starting number for sequential naming (default: 1)')

    args = parser.parse_args()

    convert_and_rename_videos(args.input_dir, args.start, args.output_dir)

"""
> Example usage:

# Basic usage - convert all videos in a directory and replace them
python fixervideo.py --input_dir /path/to/videos

# Save converted videos to a different directory
python fixervideo.py --input_dir /path/to/videos --output_dir /path/to/output

# Start numbering from a specific number
python fixervideo.py --input_dir /path/to/videos --start 100
"""