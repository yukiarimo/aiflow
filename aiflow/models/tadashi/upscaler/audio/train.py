import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
import torchaudio.transforms as T
import torch.nn.functional as F
from model import AudioUpsampler
from utils import AudioUpsampleDataset

# Define STFT Loss Helper Class (Modified for Frequency Weighting)
class STFTLoss(nn.Module):
    """Helper class for single STFT loss calculation with frequency weighting."""
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024,
                 window_fn=torch.hann_window, target_sr=48000, low_res_sr=12000,
                 high_freq_weight=1.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.target_sr = target_sr
        self.low_res_sr = low_res_sr
        self.high_freq_weight = high_freq_weight

        # Module for STFT calculation
        self.stft = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=window_fn, # Pass the original window function generator
            power=None # Get complex output
        )

        # Calculate the frequency bin corresponding to the Nyquist of the low-res signal
        self.cutoff_freq_hz = low_res_sr / 2.0
        # Bin index = Freq * FFT_size / SampleRate
        self.cutoff_bin_index = int(self.cutoff_freq_hz * self.n_fft / self.target_sr)
        print(f"  [STFTLoss n_fft={n_fft}] Weighting frequencies above {self.cutoff_freq_hz:.0f} Hz (bin ~{self.cutoff_bin_index}) with factor {self.high_freq_weight:.2f}")


    def forward(self, y_hat, y):
        """Calculate frequency-weighted STFT loss.
        Args:
            y_hat (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).
        Returns:
            Tensor: Weighted spectral convergence loss value.
            Tensor: Weighted log STFT magnitude loss value.
        """
        stft_hat = self.stft(y_hat)
        stft_y = self.stft(y)

        # Calculate magnitudes (using complex output)
        # Add epsilon for numerical stability before sqrt and log
        mag_hat = torch.sqrt(stft_hat.real.pow(2) + stft_hat.imag.pow(2) + 1e-9)
        mag_y = torch.sqrt(stft_y.real.pow(2) + stft_y.imag.pow(2) + 1e-9)

        # --- Create Frequency Weighting Mask ---
        n_freq_bins = mag_y.shape[-2] # Spectrogram shape is [B, Freq, Time]
        device = mag_y.device
        weights = torch.ones_like(mag_y) # Start with weights of 1 for all bins/times

        # Apply higher weight to bins above the low-resolution Nyquist frequency
        if self.high_freq_weight != 1.0 and self.cutoff_bin_index < n_freq_bins:
            weights[:, self.cutoff_bin_index:, :] = self.high_freq_weight

        # --- Calculate Weighted Losses ---
        # Weighted Spectral Convergence Loss
        # ||W * (M_y - M_hat)||_F / ||W * M_y||_F
        sc_loss = torch.norm(weights * (mag_y - mag_hat), p="fro") / (torch.norm(weights * mag_y, p="fro") + 1e-9)

        # Weighted Log STFT Magnitude Loss (L1)
        # L1( W * log(M_hat), W * log(M_y) )
        log_mag_hat = torch.log(mag_hat)
        log_mag_y = torch.log(mag_y)
        # Ensure weights are broadcast correctly if needed (should be automatic here)
        mag_loss = F.l1_loss(weights * log_mag_hat, weights * log_mag_y)

        return sc_loss, mag_loss

# Define Multi-Resolution STFT Loss Class (Modified to pass parameters)
class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss module with frequency weighting."""
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240], window="hann_window",
                 factor_sc=0.1, factor_mag=0.9, # Factors for combining SC and Mag within each resolution
                 target_sr=48000, low_res_sr=12000, high_freq_weight=1.0): # SR and weighting params
        """Initialize MultiResolutionSTFTLoss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor_sc (float): Weight for spectral convergence loss component.
            factor_mag (float): Weight for log STFT magnitude loss component.
            target_sr (int): Target sample rate.
            low_res_sr (int): Low resolution sample rate (for cutoff calculation).
            high_freq_weight (float): Weight factor for high frequencies.
        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        try:
             self.window_fn = getattr(torch, window)
        except AttributeError:
             raise ValueError(f"Window function '{window}' not found in torch")

        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

        # Use ModuleList to properly register the submodules
        print("Initializing MultiResolutionSTFTLoss:")
        self.stft_losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, hs, wl, self.window_fn,
                                              target_sr, low_res_sr, high_freq_weight))

    def forward(self, y_hat, y):
        """Calculate forward propagation.
        Args:
            y_hat (Tensor): Predicted signal (B, T). Note: expects [B, Length]
            y (Tensor): Ground truth signal (B, T). Note: expects [B, Length]
        Returns:
            Tensor: Combined spectral convergence and magnitude loss value, averaged over resolutions.
            Tensor: Average spectral convergence loss component.
            Tensor: Average log STFT magnitude loss component.
        """
        total_sc_loss = 0.0
        total_mag_loss = 0.0
        for loss_module in self.stft_losses:
            sc_l, mag_l = loss_module(y_hat, y)
            total_sc_loss += sc_l
            total_mag_loss += mag_l

        avg_sc_loss = total_sc_loss / len(self.stft_losses)
        avg_mag_loss = total_mag_loss / len(self.stft_losses)

        # Combine the two components here, weighted by factors, averaged over resolutions
        total_combined_loss = (self.factor_sc * avg_sc_loss) + (self.factor_mag * avg_mag_loss)

        return total_combined_loss, avg_sc_loss, avg_mag_loss # Return combined and individual components

# Main Training Function
def train(config_path, resume_checkpoint=None):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # --- Extract relevant config values ---
    cfg_data = config['data']
    cfg_audio = config['audio']
    cfg_model = config['model']
    cfg_train = config['training']

    # Create output directory
    output_dir = cfg_train['output_dir']
    chkpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        pin_memory = False # MPS doesn't support pin_memory well
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True
        print("Warning: MPS not found, using CUDA device.")
    else:
        device = torch.device("cpu")
        pin_memory = False
        print("Warning: MPS or CUDA not found, using CPU device.")

    # Prepare Dataset and DataLoader
    print("Loading training data...")
    train_dataset = AudioUpsampleDataset(
        data_dir=cfg_data.get('train_dir'),
        target_sr=cfg_audio['target_sr'],
        low_res_sr=cfg_audio['low_res_sr'],
        chunk_size_ms=cfg_audio['chunk_size_ms'],
        extensions=cfg_data['extensions']
    )
    if len(train_dataset) == 0:
         print("Error: Training dataset is empty or data dir not found. Exiting.")
         return

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg_train['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory
    )
    print(f"Dataloader initialized with {len(train_loader)} batches.")

    # --- Initialize Model ---
    upsample_factor = cfg_audio['target_sr'] // cfg_audio['low_res_sr']
    model = AudioUpsampler(
        in_channels=cfg_audio['channels'],
        out_channels=cfg_audio['channels'],
        base_channels=cfg_model['base_channels'],
        depth=cfg_model['depth'],
        upsample_factor=upsample_factor
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params:,} trainable parameters.")

    # --- Loss Functions ---
    criterion_l1 = nn.L1Loss().to(device)

    # Get STFT loss parameters from config - including the new high_freq_weight
    stft_weight = cfg_train['stft_loss_weight']
    l1_weight = cfg_train['l1_loss_weight']
    high_freq_weight = cfg_train.get('high_freq_weight', 1.0) # Default to 1.0 if not in config

    print(f"Using L1 loss (weight: {l1_weight})")
    print(f"Using MultiResSTFT loss (overall weight: {stft_weight}, internal HF weight: {high_freq_weight})")

    criterion_stft = MultiResolutionSTFTLoss(
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[512, 256, 128],
        win_lengths=[1024, 512, 256],
        # STFT Loss Factors (SC vs Mag) are within the class now (defaults are 0.1, 0.9)
        target_sr=cfg_audio['target_sr'],
        low_res_sr=cfg_audio['low_res_sr'],
        high_freq_weight=high_freq_weight # Pass the HF weight here
    ).to(device)

    # --- Optimizer and Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=cfg_train['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        patience=cfg_train['scheduler_patience'],
        factor=cfg_train['scheduler_factor'],
        verbose=True # Print message when LR decreases
    )

    start_epoch = 1
    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        try:
            ckpt = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # Handle potential KeyError if scheduler state wasn't saved previously
            if 'scheduler_state_dict' in ckpt:
                 scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            else:
                 print("Warning: Scheduler state not found in checkpoint. Initializing scheduler.")
            # Handle potential KeyError if epoch wasn't saved
            if 'epoch' in ckpt:
                 start_epoch = ckpt['epoch'] + 1
            else:
                 print("Warning: Epoch number not found in checkpoint. Starting from epoch 1.")
            print(f"Resuming from checkpoint '{resume_checkpoint}', starting at epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 1 # Reset epoch if loading fails


    # --- Training Loop ---
    print(f"Starting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, cfg_train['num_epochs'] + 1):
        model.train()
        total_combined_loss = 0.0
        total_l1_loss = 0.0
        total_stft_loss = 0.0
        total_sc_loss = 0.0 # Keep track of components
        total_mag_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg_train['num_epochs']}")

        for i, (lr_batch, hr_batch) in enumerate(pbar):
            lr_batch = lr_batch.to(device) # [B, C, L_low]
            hr_batch = hr_batch.to(device) # [B, C, L_high] Target

            optimizer.zero_grad()

            # Forward pass
            hr_pred = model(lr_batch) # [B, C, L_high] Prediction

            # Calculate losses
            loss_l1 = criterion_l1(hr_pred, hr_batch)

            # STFT Loss expects [B, T], so squeeze channel dim if mono
            # Note: For stereo, this calculates loss per channel and MultiResSTFT averages them.
            # If stereo quality is poor, consider channel averaging *before* STFT or other stereo handling.
            if hr_pred.shape[1] == 1 :
                 hr_pred_stft = hr_pred.squeeze(1)
                 hr_batch_stft = hr_batch.squeeze(1)
            else:
                 # Reshape [B, C, L] to [B*C, L] for STFT loss
                 hr_pred_stft = hr_pred.view(-1, hr_pred.shape[-1])
                 hr_batch_stft = hr_batch.view(-1, hr_batch.shape[-1])

            # loss_stft is the combination of sc and mag based on factors inside the class
            loss_stft, sc_loss, mag_loss = criterion_stft(hr_pred_stft, hr_batch_stft)

            # Combine overall losses with weights from config
            combined_loss = (l1_weight * loss_l1) + (stft_weight * loss_stft)

            # Backward pass and optimize
            combined_loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses for logging
            total_combined_loss += combined_loss.item()
            total_l1_loss += loss_l1.item()
            total_stft_loss += loss_stft.item()
            total_sc_loss += sc_loss.item() # Average SC component from MultiResSTFT
            total_mag_loss += mag_loss.item() # Average Mag component from MultiResSTFT

            # Update progress bar
            pbar.set_postfix({
                'L_Comb': f"{combined_loss.item():.4f}",
                'L1': f"{loss_l1.item():.4f}",
                'STFT': f"{loss_stft.item():.4f}" ,
                # Add components if desired:
                # 'SC': f"{sc_loss.item():.4f}",
                # 'Mag': f"{mag_loss.item():.4f}"
            })

        # Calculate average losses for the epoch
        num_batches = len(train_loader)
        avg_combined_loss = total_combined_loss / num_batches
        avg_l1_loss = total_l1_loss / num_batches
        avg_stft_loss = total_stft_loss / num_batches
        avg_sc_loss = total_sc_loss / num_batches
        avg_mag_loss = total_mag_loss / num_batches

        print(f"\nEpoch {epoch} Summary: Avg Combined Loss: {avg_combined_loss:.6f}, "
              f"Avg L1: {avg_l1_loss:.6f}, Avg STFT: {avg_stft_loss:.6f} "
              f"(Avg SC: {avg_sc_loss:.4f}, Avg Mag: {avg_mag_loss:.4f}), "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}") # Log learning rate

        # Step the scheduler (based on combined loss)
        scheduler.step(avg_combined_loss)

        # Save checkpoint
        if epoch % cfg_train['save_checkpoint_epochs'] == 0 or epoch == cfg_train['num_epochs']:
            checkpoint_path = os.path.join(chkpt_dir, f"model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_combined_loss,
                'config': config # Save config used for this training run
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the Audio Upsampling Model")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint .pth to resume from')
    args = parser.parse_args()
    train(args.config, resume_checkpoint=args.resume)