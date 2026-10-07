import argparse
import time
import torch
from models import COREML_CHUNK_FRAMES, HOP_LENGTH_RVC, Inference, SAMPLE_RATE, infer_rvc, infer_rvc_coreml, load_audio_16k, load_coreml_model, load_model
from utils import plot_spectrogram_with_pitch, salience_to_hz


def run_inference(model, audio, device, kind, hop_length_ms=None, batch_size=16, coreml_model=None, coreml_frames=COREML_CHUNK_FRAMES):
	if coreml_model is not None:
		salience = infer_rvc_coreml(coreml_model, audio, chunk_frames=coreml_frames)
		return salience, HOP_LENGTH_RVC, 10
	if kind == 'rvc':
		salience = infer_rvc(model, audio, device)
		hop_samples = HOP_LENGTH_RVC
		hop_ms = 10
	else:
		hop_ms = hop_length_ms if hop_length_ms is not None else 20
		hop_samples = int(hop_ms / 1000 * SAMPLE_RATE)
		seq_l = 2.55
		seg_len = int(seq_l * SAMPLE_RATE)
		seg_frames = int(seq_l * SAMPLE_RATE / hop_samples) + 1
		inferencer = Inference(model, seg_len, seg_frames, hop_samples, batch_size, device)
		_, salience = inferencer.inference(audio)
	return salience, hop_samples, hop_ms


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default=None, help='PyTorch model.pt or model.safetensors')
	parser.add_argument('--coreml', default=None, metavar='PATH', help='Core ML .mlpackage (Mac, uses ANE/GPU)')
	parser.add_argument('--coreml-frames', type=int, default=COREML_CHUNK_FRAMES, help='must match convert_coreml.py --frames')
	parser.add_argument('--audio', required=True, help='path to input wav')
	parser.add_argument('--hop_length', type=int, default=None, help='hop length in ms (paper=20, hub/rvc=10)')
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--plot', default=None, metavar='PATH', help='save spectrogram + pitch overlay')
	parser.add_argument('--plot-threshold', type=float, default=0.2, help='min salience for voiced pitch')
	args = parser.parse_args()

	if not args.model and not args.coreml:
		parser.error('provide --model or --coreml')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model, kind, coreml_model = None, 'rvc', None

	if args.coreml:
		coreml_model = load_coreml_model(args.coreml)
		backend = 'coreml'
	else:
		model, kind = load_model(args.model, device)
		backend = kind

	audio_np = load_audio_16k(args.audio, SAMPLE_RATE)
	audio = torch.from_numpy(audio_np).float()
	t0 = time.perf_counter()
	salience, hop_samples, hop_ms = run_inference(model, audio, device, kind, args.hop_length, args.batch_size, coreml_model=coreml_model, coreml_frames=args.coreml_frames, )
	elapsed = time.perf_counter() - t0
	freq = salience_to_hz(salience.cpu().numpy(), thred=args.plot_threshold)
	print('backend:', backend, 'hop_ms:', hop_ms, 'time:', f'{elapsed:.3f}s')
	print('frames:', len(freq))
	print('sample frequencies (Hz):', freq[:20])

	if args.plot:
		plot_spectrogram_with_pitch(audio_np, freq, hop_samples, args.plot, sr=SAMPLE_RATE, thred=args.plot_threshold, )
		print('saved plot:', args.plot)


if __name__ == '__main__':
	main()
