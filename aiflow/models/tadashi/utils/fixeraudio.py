import os
import argparse
import glob
import subprocess
import shutil
import soundfile as sf

def convert_and_rename_audio(input_dir, start_number=1, output_dir=None):
    """
    Convert all audio files in a directory to WAV format with 48 kHz sample rate, 
    clear metadata, and rename them sequentially.
    
    Args:
        input_dir (str): Directory containing audio files to process
        start_number (int): Starting number for sequential renaming
        output_dir (str): Optional output directory for converted audio
    """
    # Check if ffmpeg is installed
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg is not installed. Please install it to convert audio files.")
        return
    
    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define target directory for saving
    target_dir = output_dir if output_dir else input_dir
    
    # Find all audio files in the input directory
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.aac', '*.m4a', '*.wma', '*.aiff', '*.alac']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(input_dir, ext)))
        audio_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each audio file
    count = start_number
    processed_count = 0
    for audio_path in sorted(audio_files):
        try:
            # Skip if it's already a WAV with the correct name format
            filename = os.path.basename(audio_path)
            
            if filename == f"{count}.wav":
                # Check if it's already at 48kHz with no metadata
                try:
                    info = sf.info(audio_path)
                    if info.samplerate == 48000:
                        count += 1
                        continue
                except Exception:
                    # If we can't check the sample rate, process it anyway
                    pass
            
            # Create new filename
            new_filename = f"{count}.wav"
            new_path = os.path.join(target_dir, new_filename)
            
            # Use ffmpeg to convert to 48kHz WAV and strip metadata
            print(f"Processing: {audio_path} → {new_path}")
            subprocess.run([
                'ffmpeg',
                '-i', audio_path,
                '-ar', '48000',           # Set sample rate to 48kHz
                '-map_metadata', '-1',    # Remove metadata
                '-c:a', 'pcm_s16le',      # 16-bit PCM audio
                '-y',                     # Overwrite output without asking
                new_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"Converted and renamed: {audio_path} → {new_path}")
            processed_count += 1
            
            # Delete original if it's different and in the same directory
            if output_dir is None and os.path.abspath(audio_path) != os.path.abspath(new_path):
                os.remove(audio_path)
                print(f"Deleted original: {audio_path}")
            
            count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error converting {audio_path}: {e}")
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    print(f"Conversion complete. Processed {processed_count} audio files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert audio to 48kHz WAV, clear metadata, and rename sequentially')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, help='Optional output directory (if not specified, input_dir is used)')
    parser.add_argument('--start', type=int, default=1, help='Starting number for sequential naming (default: 1)')
    
    args = parser.parse_args()
    
    convert_and_rename_audio(args.input_dir, args.start, args.output_dir)

"""
> Example usage:

# Basic usage - convert all audio files in a directory and replace them
python fixeraudio.py --input_dir /path/to/audio

# Save converted audio to a different directory
python fixeraudio.py --input_dir /path/to/audio --output_dir /path/to/output

# Start numbering from a specific number
python fixeraudio.py --input_dir /path/to/audio --start 100
"""