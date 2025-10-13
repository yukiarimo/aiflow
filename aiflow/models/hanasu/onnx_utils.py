import torch
from onnxruntime.quantization import quantize_dynamic
from pathlib import Path
import argparse
from aiflow.models.hanasu import utils
from aiflow.models.hanasu.models import SynthesizerTrn
from aiflow.models.hanasu.text import symbols

def export_model(config_path, model_path, output_path, device='cpu'):
    """
    Export the model to ONNX in FP32 format.
    """
    print("Loading model...")
    hps = utils.get_hparams_from_file(config_path)
    model_g = SynthesizerTrn(
        len(symbols),
        hps.data.n_mel_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    ).to(device)

    utils.load_checkpoint(model_path, model_g, None)
    model_g.eval()
    model_g = model_g.float()
    for param in model_g.parameters(): param.data = param.data.float()
    for buffer in model_g.buffers(): buffer.data = buffer.data.float()
    print("Model loaded.")
    model_g.forward = model_g.infer

    # Prepare dummy inputs for tracing
    sequences = torch.randint(low=0, high=len(symbols), size=(1, 50), dtype=torch.long).to(device)
    sequence_lengths = torch.LongTensor([sequences.size(1)]).to(device)
    sid_input = torch.LongTensor([0]).to(device) if hps.data.n_speakers > 0 else None

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_onnx_path = output_file.with_suffix(".temp.onnx")

    print(f"Exporting to ONNX at {temp_onnx_path}...")
    torch.onnx.export(
        model=model_g,
        args=(sequences, sequence_lengths),
        f=str(temp_onnx_path),
        verbose=False,
        opset_version=17,
        input_names=["x", "x_lengths"],
        output_names=["output", "attn", "y_mask", "internals"],
        dynamic_axes={"x": {0: "batch_size", 1: "phonemes"}, "x_lengths": {0: "batch_size"}, "output": {0: "batch_size", 2: "time"}},
        kwargs={'sid': sid_input, 'noise_scale': 0.667, 'length_scale': 1.0, 'noise_scale_w': 0.8}
    )
    print("Initial ONNX export complete.")
    temp_onnx_path.rename(output_file)
    print(f"FP32 model saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Hanasu model to optimized ONNX format.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model config.json file.")
    parser.add_argument("--model", type=str, required=True, help="Path to the PyTorch model checkpoint (.pth).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output ONNX model.")
    args = parser.parse_args()
    export_model(args.config, args.model, args.output)