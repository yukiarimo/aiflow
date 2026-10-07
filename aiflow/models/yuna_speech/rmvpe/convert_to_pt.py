import argparse
import os
import torch
from models import load_model


def main():
	parser = argparse.ArgumentParser(description='Convert safetensors hub weights to model.pt')
	parser.add_argument('--input', default='model/model.safetensors', help='path to .safetensors')
	parser.add_argument('--output', default=None, help='output .pt path (default: same dir as input)')
	args = parser.parse_args()
	out = args.output or os.path.join(os.path.dirname(args.input), 'model.pt')
	model, kind = load_model(args.input, device='cpu')
	torch.save({'state_dict': model.state_dict(), 'kind': kind}, out)
	print('saved', out, 'arch=', kind)


if __name__ == '__main__':
	main()
