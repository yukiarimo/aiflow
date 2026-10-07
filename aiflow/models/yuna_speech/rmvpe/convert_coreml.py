import argparse
import json
import os
import torch
from models import COREML_CHUNK_FRAMES, load_model


def main():
	parser = argparse.ArgumentParser(description='Convert RMVPE (RVC hub) to Core ML')
	parser.add_argument('--input', default='model/model.pt', help='model.pt or model.safetensors')
	parser.add_argument('--output', default=None, help='output .mlpackage path')
	parser.add_argument('--frames', type=int, default=COREML_CHUNK_FRAMES, help='mel frames per CoreML call (multiple of 32)')
	args = parser.parse_args()

	if args.frames % 32 != 0:
		raise SystemExit('--frames must be a multiple of 32')

	try:
		import coremltools as ct
	except ImportError:
		raise SystemExit('Install coremltools: pip install coremltools')

	model, kind = load_model(args.input, device='cpu')
	if kind != 'rvc':
		raise SystemExit('Core ML export only supports hub/RVC checkpoints (use model.pt / model.safetensors).')

	model.eval()
	example = torch.randn(1, 128, args.frames)
	with torch.no_grad():
		traced = torch.jit.trace(model, example, strict=False)

	mlmodel = ct.convert(traced, inputs=[ct.TensorType(name='mel', shape=example.shape)], outputs=[ct.TensorType(name='salience')], convert_to='mlprogram', minimum_deployment_target=ct.target.macOS13, )
	out = args.output or os.path.join(os.path.dirname(args.input), 'rmvpe.mlpackage')
	mlmodel.save(out)
	meta_path = os.path.join(os.path.dirname(os.path.abspath(out)), 'rmvpe_coreml.meta.json')
	with open(meta_path, 'w') as f:
		json.dump({'arch': 'rvc', 'chunk_frames': args.frames, 'hop_length': 160, 'n_mels': 128}, f, indent=2)

	print('saved', out)
	print('meta', meta_path)
	print('infer: python infer.py --coreml', out, '--audio your.wav')


if __name__ == '__main__':
	main()
