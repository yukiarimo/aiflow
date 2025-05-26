import torch
import torchaudio
import torchaudio.transforms as T
import os
import random
from torch.utils.data import Dataset
import glob
import torch.nn.functional as F # Added for F.interpolate

def load_audio(path, target_sr):
    """Loads audio, resamples, converts to mono, normalizes."""
    try:
        waveform, sr = torchaudio.load(path)
        # Resample if needed
        if sr != target_sr:
            # Use interpolation mode that's often good for downsampling/upsampling
            resampler = T.Resample(sr, target_sr, resampling_method='sinc_interp_kaiser')
            waveform = resampler(waveform)
        return waveform
    except Exception as e:
        print(f"Error loading audio file {path}: {e}")
        return None

def create_data_pair(waveform_hr, target_sr, low_res_sr):
    """Creates low-resolution version from high-resolution waveform."""
    resample_lr = T.Resample(target_sr, low_res_sr, resampling_method='sinc_interp_kaiser')
    waveform_lr = resample_lr(waveform_hr)
    return waveform_lr, waveform_hr

class AudioUpsampleDataset(Dataset):
    def __init__(self, data_dir, target_sr, low_res_sr, chunk_size_ms, extensions):
        self.target_sr = target_sr
        self.low_res_sr = low_res_sr
        self.chunk_len_hr = int(target_sr * chunk_size_ms / 1000)
        self.chunk_len_lr = int(low_res_sr * chunk_size_ms / 1000)
        self.upsample_factor = target_sr // low_res_sr

        self.file_paths = []
        if data_dir and os.path.isdir(data_dir):
            for ext in extensions:
                self.file_paths.extend(glob.glob(os.path.join(data_dir, f'**/*{ext}'), recursive=True))
            print(f"Found {len(self.file_paths)} files in {data_dir}")
            if not self.file_paths:
                 print(f"Warning: No audio files found in {data_dir} with extensions {extensions}")
        else:
             print(f"Warning: Data directory '{data_dir}' not found or not specified.")

        self.cache = {} # Simple cache for loaded files
        self.min_len_hr = self.chunk_len_hr # Minimum length required for a file to be used

    def __len__(self):
        # Rough estimate, better to just use a large number if dataset is large enough
        # or iterate through files if dataset is small
        if not self.file_paths:
             return 0
        # Estimate based on average file length (could be refined)
        # Let's estimate roughly 5 chunks per file on average that are long enough
        estimated_chunks = len(self.file_paths) * 5
        return max(1, estimated_chunks) # Ensure it's at least 1

    def __getitem__(self, idx):
         if not self.file_paths:
             raise IndexError("No audio files available in the dataset.")

         # Try multiple times to find a suitable file/chunk
         for _ in range(10): # Limit attempts to avoid infinite loops
            file_path = random.choice(self.file_paths)

            waveform_hr = self.cache.get(file_path)
            if waveform_hr is None:
                waveform_hr = load_audio(file_path, self.target_sr)
                if waveform_hr is None or waveform_hr.shape[1] < self.min_len_hr:
                     # If loading failed or file too short, discard from cache (if present) and try again
                     if file_path in self.cache: del self.cache[file_path]
                     continue
                self.cache[file_path] = waveform_hr
                # Limit cache size if needed (e.g., simple FIFO or LRU)
                if len(self.cache) > 5: # Keep roughly 5 files in cache
                    self.cache.pop(next(iter(self.cache)))

            # If file is exactly chunk size, use it directly
            if waveform_hr.shape[1] == self.chunk_len_hr:
                chunk_hr = waveform_hr
            # If file is longer, take a random chunk
            elif waveform_hr.shape[1] > self.chunk_len_hr:
                start_idx = random.randint(0, waveform_hr.shape[1] - self.chunk_len_hr - 1)
                chunk_hr = waveform_hr[:, start_idx : start_idx + self.chunk_len_hr]
            # This case should ideally not be reached due to the min_len check, but handle defensively
            else:
                 continue # Try another file

            # Create low-resolution version
            chunk_lr, chunk_hr_target = create_data_pair(chunk_hr, self.target_sr, self.low_res_sr)

            # Ensure LR chunk has the expected length, pad/trim if needed (can happen due to resampling)
            current_lr_len = chunk_lr.shape[1]
            if current_lr_len < self.chunk_len_lr:
                padding = self.chunk_len_lr - current_lr_len
                chunk_lr = torch.nn.functional.pad(chunk_lr, (0, padding))
            elif current_lr_len > self.chunk_len_lr:
                chunk_lr = chunk_lr[:, :self.chunk_len_lr]

            return chunk_lr, chunk_hr_target # Return LR input and HR target

         # If loop finishes without returning, something is wrong
         raise RuntimeError(f"Could not get a valid chunk after 10 attempts. Check data and chunk sizes.")

# Helper for inference chunking with improved overlap-add
def process_in_chunks(model, waveform_lr, chunk_len_lr, upsample_factor, device, overlap_ratio=0.5):
    """Process long audio in overlapping chunks for inference with Hanning window."""
    model.eval()
    in_channels, total_len_lr = waveform_lr.shape
    target_total_len_hr = total_len_lr * upsample_factor
    chunk_len_hr = chunk_len_lr * upsample_factor

    # Calculate step sizes and overlap lengths for low and high resolution
    step_lr = int(chunk_len_lr * (1 - overlap_ratio))
    if step_lr <= 0: step_lr = chunk_len_lr # Ensure step is positive, avoid infinite loop if overlap is 1.0
    step_hr = step_lr * upsample_factor
    overlap_len_hr = chunk_len_hr - step_hr

    # Use reflect padding for smoother boundaries
    # Padding length should accommodate half a chunk on each side for the window overlap
    pad_len_lr = chunk_len_lr // 2
    padded_waveform_lr = F.pad(waveform_lr, (pad_len_lr, pad_len_lr), mode='reflect').to(device)
    _, padded_len_lr = padded_waveform_lr.shape

    # Prepare output buffer and sum-of-windows buffer on CPU
    output_waveform_hr = torch.zeros((in_channels, target_total_len_hr), device='cpu')
    window_sum = torch.zeros((in_channels, target_total_len_hr), device='cpu')

    # Create Hanning window for smooth overlap-add
    # Use periodic=False as the window is applied and summed N times
    hann_window = torch.hann_window(chunk_len_hr, periodic=False).unsqueeze(0).to('cpu') # [1, chunk_len_hr]

    with torch.no_grad():
        idx_lr = 0
        while idx_lr < padded_len_lr:
            # Define current chunk boundaries
            start_lr = idx_lr
            end_lr = start_lr + chunk_len_lr

            # Get the input chunk
            chunk_lr = padded_waveform_lr[:, start_lr:end_lr]
            actual_chunk_len_lr = chunk_lr.shape[1]

            # Pad if the chunk is too short (at the very end of the padded signal)
            if actual_chunk_len_lr < chunk_len_lr:
                padding = chunk_len_lr - actual_chunk_len_lr
                chunk_lr = F.pad(chunk_lr, (0, padding), mode='constant', value=0)

            # Add batch dimension for the model
            chunk_lr_batch = chunk_lr.unsqueeze(0)

            # Perform upsampling
            chunk_hr_pred = model(chunk_lr_batch) # [1, C, chunk_len_hr]
            chunk_hr_pred = chunk_hr_pred.squeeze(0).cpu() # [C, chunk_len_hr]

            # --- Overlap-Add Step ---
            # Calculate corresponding high-resolution indices for placing this chunk
            start_hr = (idx_lr - pad_len_lr) * upsample_factor # Position in original output coordinates
            end_hr = start_hr + chunk_len_hr

            # Define boundaries clamping to the valid output range [0, target_total_len_hr]
            out_start_hr = max(0, start_hr)
            out_end_hr = min(target_total_len_hr, end_hr)

            # Define boundaries for slicing the *predicted chunk* and the *window*
            # Handles cases where the chunk calculation goes beyond the actual output length
            pred_start_idx = out_start_hr - start_hr
            pred_end_idx = pred_start_idx + (out_end_hr - out_start_hr)

            # Ensure indices are valid
            if out_start_hr < out_end_hr and pred_start_idx < pred_end_idx:
                 # Add the windowed chunk to the output buffer
                 output_waveform_hr[:, out_start_hr:out_end_hr] += \
                     chunk_hr_pred[:, pred_start_idx:pred_end_idx] * hann_window[:, pred_start_idx:pred_end_idx]

                 # Add the window itself to the window sum buffer
                 window_sum[:, out_start_hr:out_end_hr] += hann_window[:, pred_start_idx:pred_end_idx]

            # Move to the next chunk start position
            # Break if next step would start beyond the padded length needed to generate the last samples
            if start_lr + step_lr >= padded_len_lr :
                break
            idx_lr += step_lr

    # Avoid division by zero: replace near-zero window sums with 1.
    # The window sum should ideally be > 0 where there is signal.
    window_sum = torch.where(window_sum < 1e-6, torch.tensor(1.0, device='cpu'), window_sum)

    # Normalize the output by the sum of windows
    final_output_hr = output_waveform_hr / window_sum

    # Clamp final output to [-1, 1] range
    final_output_hr = torch.clamp(final_output_hr, -1.0, 1.0)

    return final_output_hr