import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

def get_padding(k, d):
    return int((k * d - d) / 2)

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig

def save_checkpoint(
    checkpoint_dir,
    generator,
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    step,
    loss,
    best,
    logger,
):
    state = {
        "generator": {
            "model": generator.state_dict(),
            "optimizer": optimizer_generator.state_dict(),
            "scheduler": scheduler_generator.state_dict(),
        },
        "discriminator": {
            "model": discriminator.state_dict(),
            "optimizer": optimizer_discriminator.state_dict(),
            "scheduler": scheduler_discriminator.state_dict(),
        },
        "step": step,
        "loss": loss,
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f"model-{step}.pt"
    torch.save(state, checkpoint_path)
    if best:
        best_path = checkpoint_dir / "model-best.pt"
        torch.save(state, best_path)
    print(f"Saved checkpoint: {checkpoint_path.stem}")

def load_checkpoint(
    load_path,
    generator,
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    rank,
    finetune=False,
):
    print(f"Loading checkpoint from {load_path}")

    # Handle device mapping correctly for both CPU and CUDA
    if rank >= 0:  # CUDA device
        map_location = {"cuda:0": f"cuda:{rank}"}
    else:  # CPU or other device
        map_location = "cpu"

    checkpoint = torch.load(load_path, map_location=map_location)

    # Get the generator state dict
    generator_state = checkpoint["generator"]["model"]

    # Remove the 'module.' prefix from keys
    fixed_generator_state = {}
    for key, value in generator_state.items():
        if key.startswith("module."):
            fixed_generator_state[key[7:]] = value  # Remove 'module.' (7 characters)
        else:
            fixed_generator_state[key] = value

    # Load the modified state dict
    generator.load_state_dict(fixed_generator_state)

    if discriminator is not None and "discriminator" in checkpoint:
        discriminator_state = checkpoint["discriminator"]["model"]

        # Remove the 'module.' prefix from discriminator keys too
        fixed_discriminator_state = {}
        for key, value in discriminator_state.items():
            if key.startswith("module."):
                fixed_discriminator_state[key[7:]] = value
            else:
                fixed_discriminator_state[key] = value

        discriminator.load_state_dict(fixed_discriminator_state)

    if not finetune:
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint["generator"]["optimizer"])
        if scheduler_generator is not None:
            scheduler_generator.load_state_dict(checkpoint["generator"]["scheduler"])
        if optimizer_discriminator is not None and "discriminator" in checkpoint:
            optimizer_discriminator.load_state_dict(
                checkpoint["discriminator"]["optimizer"]
            )
        if scheduler_discriminator is not None and "discriminator" in checkpoint:
            scheduler_discriminator.load_state_dict(
                checkpoint["discriminator"]["scheduler"]
            )

    return checkpoint["step"], checkpoint["loss"]

def hifigan(checkpoint_path="model.pth", map_location=None, **kwargs):
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
    from .generator import HifiganGenerator # Use relative import
    hifigan = HifiganGenerator() # Assumes default HifiganGenerator params are okay
    ckpt_path = checkpoint_path
    print(f"Loading HiFi-GAN checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=map_location)

    # Determine the correct state dict key
    if "generator" in checkpoint and "model" in checkpoint["generator"]:
        # Structure from hifigan/train.py save_checkpoint
        generator_state = checkpoint["generator"]["model"]
        print("Loading state_dict from checkpoint['generator']['model']")
    elif "generator" in checkpoint and isinstance(checkpoint["generator"], dict) and "state_dict" in checkpoint["generator"]:
        # Alternative common pattern
        generator_state = checkpoint["generator"]["state_dict"]
        print("Loading state_dict from checkpoint['generator']['state_dict']")
    elif "generator" in checkpoint:
            # If checkpoint['generator'] is the state_dict itself
            generator_state = checkpoint["generator"]
            print("Loading state_dict from checkpoint['generator']")
    elif "model" in checkpoint:
            # If checkpoint['model'] is the state_dict
            generator_state = checkpoint["model"]
            print("Loading state_dict from checkpoint['model']")
    elif isinstance(checkpoint, dict) and any(k.startswith("conv_pre") for k in checkpoint.keys()):
            # If checkpoint itself is the state_dict
            generator_state = checkpoint
            print("Loading state_dict directly from checkpoint")
    else:
            raise KeyError("Could not find generator state_dict in checkpoint. Keys found:", list(checkpoint.keys()))

    # Remove module prefix if present (happens with DataParallel/DDP models)
    consume_prefix_in_state_dict_if_present(generator_state, "module.")

    # Load the state dict
    missing_keys, unexpected_keys = hifigan.load_state_dict(generator_state, strict=True) # Be strict for vocoder
    if missing_keys:
        print(f"Warning: Missing keys in HiFi-GAN state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in HiFi-GAN state_dict: {unexpected_keys}")

    hifigan.eval()
    hifigan.remove_weight_norm()
    print("Successfully loaded HiFi-GAN model weights.")
    return hifigan