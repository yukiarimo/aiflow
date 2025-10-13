import os, glob, sys, argparse, logging, json, subprocess, shutil
import numpy as np
from scipy.io.wavfile import read
import torch
MATPLOTLIB_FLAG = False
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None: optimizer.load_state_dict(checkpoint_dict["optimizer"])

    saved_state_dict = checkpoint_dict["model"]
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        try:
            if k in saved_state_dict and saved_state_dict[k].shape == v.shape: new_state_dict[k] = saved_state_dict[k]
            else:
                # Log message for mismatched shapes or missing keys
                if k in saved_state_dict: print(f"NOTICE: Size mismatch for {k}. Expected {v.shape}, got {saved_state_dict[k].shape}. Using model initialization.")
                else: print(f"{k} is not in the checkpoint")
                new_state_dict[k] = v
        except:
            print(f"{k} is not in the checkpoint")
            new_state_dict[k] = v

    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(new_state_dict)
    print(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
    return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    print(f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}")
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({"model": state_dict, "iteration": iteration,"optimizer": optimizer.state_dict(), "learning_rate": learning_rate}, checkpoint_path)

def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=48000):
    for k, v in scalars.items(): writer.add_scalar(k, v, global_step)
    for k, v in histograms.items(): writer.add_histogram(k, v, global_step)
    for k, v in images.items(): writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items(): writer.add_audio(k, v, global_step, audio_sampling_rate)

def scan_checkpoint(dir_path, regex):
    f_list = sorted(glob.glob(os.path.join(dir_path, regex)), key=lambda f: int("".join(filter(str.isdigit, f))))
    return f_list or None

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = scan_checkpoint(dir_path, regex)
    if not f_list: return None
    x = f_list[-1]
    print(x)
    return x

def remove_old_checkpoints(cp_dir, prefixes=['G_*.pth', 'D_*.pth', 'DUR_*.pth']):
    for prefix in prefixes:
        sorted_ckpts = scan_checkpoint(cp_dir, prefix)
        if sorted_ckpts and len(sorted_ckpts) > 3:
            for ckpt_path in sorted_ckpts[:-3]:
                os.remove(ckpt_path)
                print(f"removed {ckpt_path}")

def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        MATPLOTLIB_FLAG = True
    import matplotlib.pylab as plt

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        MATPLOTLIB_FLAG = True
    import matplotlib.pylab as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None: xlabel += f"\n\n{info}"
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f: return [line.strip().split(split) for line in f]

def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./config.json", help="JSON file for configuration")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)
    os.makedirs(model_dir, exist_ok=True)
    config_save_path = os.path.join(model_dir, "config.json")
    if init: shutil.copy(args.config, config_save_path)
    with open(config_save_path, "r") as f: config = json.load(f)
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams

def get_hparams_from_dir(model_dir):
    with open(os.path.join(model_dir, "config.json"), "r") as f: config = json.load(f)
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams

def get_hparams_from_file(config_path):
    with open(config_path, "r") as f: config = json.load(f)
    return HParams(**config)

def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(f"{source_dir} is not a git repository, therefore hash value comparison will be ignored.")
        return
    cur_hash = subprocess.getoutput("git rev-parse HEAD")
    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        with open(path) as f: saved_hash = f.read()
        if saved_hash != cur_hash: logger.warn(f"git hash values are different. {saved_hash[:8]}(saved) != {cur_hash[:8]}(current)")
    else:
        with open(path, "w") as f: f.write(cur_hash)

def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    os.makedirs(model_dir, exist_ok=True)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict: v = HParams(**v)
            self[k] = v
    def keys(self): return self.__dict__.keys()
    def items(self): return self.__dict__.items()
    def values(self): return self.__dict__.values()
    def __len__(self): return len(self.__dict__)
    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, value): return setattr(self, key, value)
    def __contains__(self, key): return key in self.__dict__
    def __repr__(self): return self.__dict__.__repr__()