import argparse
import json
import os
import shutil
import struct
import time
import numpy as np
import torch
from models import COREML_CHUNK_FRAMES, ExportVAD, LiquidVAD, infer_vad_coreml, load_vad, load_vad_coreml
from utils import DEFAULT_CKPT, DEFAULT_DATA, DEFAULT_RUNS, HOP_LENGTH, N_MELS, RMVPE_DIM, SAMPLE_RATE, cache_dir, load_npz, resolve_ckpt

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
DEFAULT_ONNX = os.path.join(DEFAULT_RUNS, 'liquid_vad.onnx')
DEFAULT_ML = os.path.join(DEFAULT_RUNS, 'liquid_vad.mlpackage')
WEB_ONNX = os.path.join(WEB_DIR, 'liquid_vad.onnx')
MEL_BASIS_BIN = os.path.join(WEB_DIR, 'mel_basis.bin')
ANE_TARGET = 500.0


def _probe_mel(frames):
	path = os.path.join(cache_dir(DEFAULT_DATA), 'take_01.npz')
	if os.path.isfile(path):
		mel = load_npz(path)['mel']
		t = mel.shape[-1]
		if t < frames:
			mel = np.pad(mel, ((0, 0), (0, frames - t)))
		return torch.from_numpy(np.ascontiguousarray(mel[:, :frames])).unsqueeze(0)
	torch.manual_seed(0)
	return torch.randn(1, N_MELS, frames)


def _pair(frames, kind='zero'):
	mel = _probe_mel(frames)
	if kind == 'zero':
		rmvpe = torch.zeros(1, RMVPE_DIM, frames)
	else:
		g = torch.Generator().manual_seed(frames + 3)
		rmvpe = torch.rand(1, RMVPE_DIM, frames, generator=g)
	return mel, rmvpe


def _align(got):
	got = np.asarray(got, dtype=np.float32)
	if got.ndim == 3:
		got = got[0]
	if got.shape[0] == 3:
		return got
	return got.T


def _sigmoid_err(ref, got):
	pref = 1.0 / (1.0 + np.exp(-np.clip(ref, -20, 20)))
	pgot = 1.0 / (1.0 + np.exp(-np.clip(got, -20, 20)))
	return float(np.max(np.abs(ref - got))), float(np.max(np.abs(pref - pgot)))


def _write_meta(path, ckpt, model, extra):
	meta = {'arch': 'liquid_gate', 'fused': False, 'hop_length': HOP_LENGTH, 'n_mels': N_MELS, 'rmvpe_dim': RMVPE_DIM, 'sample_rate': SAMPLE_RATE, 'n_fft': 1024, 'win_length': 1024, 'fmin': 30.0, 'fmax': 8000.0, 'htk': True, 'inputs': ['mel', 'rmvpe'], 'outputs': ['logits'], 'layout': 'mel [1,128,T] + rmvpe [1,3,T] -> logits [1,3,T]', 'epoch': ckpt.get('epoch'), 'loss': ckpt.get('loss'), 'config': model.config(), 'pitch': 'optional input, zeros if estimation failed', }
	meta.update(extra)
	meta_path = os.path.abspath(path) + '.meta.json'
	with open(meta_path, 'w') as f:
		json.dump(meta, f, indent=2)
	return meta_path


def _build(args):
	if args.random:
		model = LiquidVAD().eval()
		ckpt = {'epoch': 0, 'loss': None, 'arch': 'liquid_gate'}
		return ExportVAD(model).eval(), ckpt, model, 'random'
	path = resolve_ckpt(args.input)
	if not os.path.isfile(path):
		raise SystemExit('no checkpoint at %s — run train.py, or pass --random' % path)
	model, ckpt = load_vad(path, device='cpu')
	return ExportVAD(model).eval(), ckpt, model, path


def _xtimes(audio_s, fn, n=12, warmup=4):
	for _ in range(warmup):
		fn()
	t0 = time.perf_counter()
	for _ in range(n):
		fn()
	dt = (time.perf_counter() - t0) / n
	return audio_s / max(dt, 1e-9), dt * 1000.0


def convert_coreml(args):
	try:
		import coremltools as ct
	except ImportError:
		raise SystemExit('Install coremltools: pip install coremltools')

	export, ckpt, model, src = _build(args)
	print('checkpoint', src)
	mel, rmvpe = _pair(args.frames, 'rand')
	with torch.no_grad():
		traced = torch.jit.trace(export, (mel, rmvpe), strict=False)
	mlmodel = ct.convert(traced, inputs=[ct.TensorType(name='mel', shape=(1, N_MELS, args.frames)), ct.TensorType(name='rmvpe', shape=(1, RMVPE_DIM, args.frames)), ], outputs=[ct.TensorType(name='logits')], convert_to='mlprogram', minimum_deployment_target=ct.target.macOS13, compute_precision=ct.precision.FLOAT16, )
	out = args.output or DEFAULT_ML
	os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
	mlmodel.save(out)
	meta_path = _write_meta(out, ckpt, model, {'format': 'coreml', 'chunk_frames': args.frames, 'flexible': False, 'compute': 'ALL', 'precision': 'FLOAT16', })
	print('saved', out)
	print('meta', meta_path)
	with torch.no_grad():
		ref = export(mel, rmvpe).numpy()[0]
	ml = load_vad_coreml(out, units=ct.ComputeUnit.ALL)
	got = _align(infer_vad_coreml(ml, mel[0].numpy(), rmvpe[0].numpy(), frames=args.frames, overlap=0))[:, :args.frames]
	logit_err, prob_err = _sigmoid_err(ref, got)
	print('max |pt-coreml| logits=%.5f  prob=%.5f  (T=%d, pitch random)' % (logit_err, prob_err, args.frames))
	zmel, zpitch = _pair(args.frames, 'zero')
	with torch.no_grad():
		zref = export(zmel, zpitch).numpy()[0]
	zgot = _align(infer_vad_coreml(ml, zmel[0].numpy(), zpitch[0].numpy(), frames=args.frames, overlap=0))[:, :args.frames]
	_, zprob = _sigmoid_err(zref, zgot)
	print('max |pt-coreml| prob=%.5f  (T=%d, pitch zeros)' % (zprob, args.frames))
	if max(prob_err, zprob) > 3e-2:
		raise SystemExit('Core ML prob drift %.4f > 0.030' % max(prob_err, zprob))
	print('Core ML package ok — mel + optional pitch')
	flex = _try_flexible(ct, traced, out, args.frames)
	return out, flex


def _try_flexible(ct, traced, fixed_path, frames):
	t = ct.RangeDim(lower_bound=16, upper_bound=8192, default=frames)
	try:
		mlmodel = ct.convert(traced, inputs=[ct.TensorType(name='mel', shape=(1, N_MELS, t)), ct.TensorType(name='rmvpe', shape=(1, RMVPE_DIM, t)), ], outputs=[ct.TensorType(name='logits')], convert_to='mlprogram', minimum_deployment_target=ct.target.macOS13, compute_precision=ct.precision.FLOAT16, )
	except Exception as e:
		print('flexible Core ML skipped:', e)
		return None
	flex = fixed_path.replace('.mlpackage', '_flex.mlpackage')
	mlmodel.save(flex)
	print('saved flexible', flex)
	print('flexible length stays on CPU. Infer loads the fixed chunk, which is the Neural Engine graph.')
	return flex


def _write_mel_basis(path):
	from librosa.filters import mel as librosa_mel
	basis = librosa_mel(sr=SAMPLE_RATE, n_fft=1024, n_mels=N_MELS, fmin=30.0, fmax=8000.0, htk=True).astype(np.float32)
	os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
	with open(path, 'wb') as f:
		f.write(struct.pack('<II', basis.shape[0], basis.shape[1]))
		f.write(basis.tobytes(order='C'))
	return basis


def _export_onnx(export, path, frames, dynamic):
	mel, rmvpe = _pair(frames, 'rand')
	os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
	dynamic_axes = {'mel': {0: 'B', 2: 'T'}, 'rmvpe': {0: 'B', 2: 'T'}, 'logits': {0: 'B', 2: 'T'}, } if dynamic else None
	kwargs = dict(input_names=['mel', 'rmvpe'], output_names=['logits'], opset_version=17, do_constant_folding=True, dynamo=False)
	if dynamic_axes:
		kwargs['dynamic_axes'] = dynamic_axes
	try:
		torch.onnx.export(export, (mel, rmvpe), path, **kwargs)
		return 'jit'
	except Exception as e:
		print('jit export failed:', e)
		kwargs['dynamo'] = True
		torch.onnx.export(export, (mel, rmvpe), path, **kwargs)
		return 'dynamo'


def _ort_run(sess, mel, rmvpe):
	feed = {'mel': np.ascontiguousarray(mel, dtype=np.float32), 'rmvpe': np.ascontiguousarray(rmvpe, dtype=np.float32), }
	got = np.asarray(sess.run(None, feed)[0], dtype=np.float32)
	return _align(got)


def _check_onnx(sess, export, frames, extra_t=None, atol_prob=3e-2):
	import onnxruntime as ort
	tests = [frames]
	if extra_t:
		tests.extend(extra_t)
	worst = 0.0
	for t in tests:
		for kind in ('zero', 'rand'):
			mel, rmvpe = _pair(t, kind)
			with torch.no_grad():
				ref = export(mel, rmvpe).numpy()[0]
			got = _ort_run(sess, mel.numpy(), rmvpe.numpy())[:, :t]
			if got.shape[-1] != t or got.shape[0] != 3:
				raise SystemExit('ONNX shape %s != [3,%d]' % (got.shape, t))
			logit_err, prob_err = _sigmoid_err(ref, got)
			print('max |pt-onnx| logits=%.5f  prob=%.5f  (T=%d, pitch %s)' % (logit_err, prob_err, t, kind))
			worst = max(worst, prob_err)
			if prob_err > atol_prob:
				raise SystemExit('ONNX prob drift %.4f > %.3f at T=%d' % (prob_err, atol_prob, t))
	print('providers', sess.get_providers())
	print('onnxruntime', ort.__version__)
	return worst


def convert_onnx(args):
	try:
		import onnxruntime as ort
	except ImportError:
		raise SystemExit('Install onnxruntime: pip install onnxruntime')

	export, ckpt, model, src = _build(args)
	print('checkpoint', src)
	out = args.output or DEFAULT_ONNX
	dynamic = not args.fixed
	kind = _export_onnx(export, out, args.frames, dynamic)
	print('saved', out, 'via', kind, 'dynamic' if dynamic else 'fixed', 'T=%d' % args.frames)
	sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
	extra = [128, 384, 2000] if dynamic else None
	try:
		_check_onnx(sess, export, args.frames, extra)
	except SystemExit as e:
		if not dynamic:
			raise
		print('dynamic check failed (%s) — retrying fixed T=%d' % (e, args.frames))
		dynamic = False
		kind = _export_onnx(export, out, args.frames, False)
		sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
		_check_onnx(sess, export, args.frames, None)
	meta_path = _write_meta(out, ckpt, model, {'format': 'onnx', 'chunk_frames': args.frames, 'dynamic': bool(dynamic), 'export': kind})
	print('meta', meta_path)
	basis = _write_mel_basis(MEL_BASIS_BIN)
	print('mel basis', MEL_BASIS_BIN, basis.shape)
	if not args.random:
		os.makedirs(WEB_DIR, exist_ok=True)
		shutil.copy2(out, WEB_ONNX)
		shutil.copy2(meta_path, os.path.join(WEB_DIR, 'liquid_vad.meta.json'))
		print('web', WEB_ONNX)
	if args.serve:
		import http.server
		import socketserver
		os.chdir(WEB_DIR)
		with socketserver.TCPServer(('127.0.0.1', args.port), http.server.SimpleHTTPRequestHandler) as httpd:
			print('open http://127.0.0.1:%d/' % args.port)
			httpd.serve_forever()
	return out


def _bench_torch(export, frames):
	mel, rmvpe = _pair(frames, 'zero')
	audio_s = frames * HOP_LENGTH / float(SAMPLE_RATE)
	rows = []
	export_cpu = export.cpu().eval()
	with torch.no_grad():
		x, speed = _xtimes(audio_s, lambda: export_cpu(mel, rmvpe))
	rows.append(('pytorch-cpu', x, speed))
	if torch.backends.mps.is_available():
		export_mps = export.to('mps').eval()
		mm, rr = mel.to('mps'), rmvpe.to('mps')

		def _mps():
			export_mps(mm, rr)
			torch.mps.synchronize()

		with torch.no_grad():
			x, speed = _xtimes(audio_s, _mps)
		rows.append(('pytorch-gpu-mps', x, speed))
		export.cpu()
	return rows


def _bench_onnx(path, frames):
	import onnxruntime as ort
	mel, rmvpe = _pair(frames, 'zero')
	audio_s = frames * HOP_LENGTH / float(SAMPLE_RATE)
	feed_m = mel.numpy()
	feed_r = rmvpe.numpy()
	rows = []
	available = ort.get_available_providers()
	print('onnx providers available', available)
	want = [('onnx-cpu', 'CPUExecutionProvider'), ('onnx-coreml', 'CoreMLExecutionProvider')]
	for label, provider in want:
		if provider not in available:
			print('skip', label)
			continue
		sess = ort.InferenceSession(path, providers=[provider])

		def _run(sess=sess):
			_ort_run(sess, feed_m, feed_r)

		x, ms = _xtimes(audio_s, _run)
		rows.append((label, x, ms))
	return rows


def _bench_coreml(path, frames, long_frames=3072):
	import coremltools as ct
	mel, rmvpe = _pair(frames, 'zero')
	audio_s = frames * HOP_LENGTH / float(SAMPLE_RATE)
	m = mel[0].numpy()
	r = rmvpe[0].numpy()
	rows = []
	units = (('coreml-cpu', ct.ComputeUnit.CPU_ONLY), ('coreml-gpu', ct.ComputeUnit.CPU_AND_GPU), ('coreml-ane', ct.ComputeUnit.CPU_AND_NE), ('coreml-all', ct.ComputeUnit.ALL), )
	for label, unit in units:
		ml = load_vad_coreml(path, units=unit)
		x, ms = _xtimes(audio_s, lambda ml=ml: infer_vad_coreml(ml, m, r, frames=frames, overlap=0))
		rows.append((label, x, ms))
	long_s = long_frames * HOP_LENGTH / float(SAMPLE_RATE)
	lm, lr = _pair(long_frames, 'zero')
	ml = load_vad_coreml(path, units=ct.ComputeUnit.CPU_AND_NE)
	x, ms = _xtimes(long_s, lambda: infer_vad_coreml(ml, lm[0].numpy(), lr[0].numpy(), frames=frames), n=6, warmup=2)
	rows.append(('coreml-ane-long', x, ms))
	return rows


def bench(export, onnx_path, ml_path, frames):
	print('--- speed (× real time, higher is faster) ---')
	rows = []
	rows.extend(_bench_torch(export, frames))
	if onnx_path:
		rows.extend(_bench_onnx(onnx_path, frames))
	if ml_path:
		rows.extend(_bench_coreml(ml_path, frames))
	ane = [r for r in rows if r[0] in ('coreml-ane', 'coreml-ane-long', 'coreml-all')]
	for name, factor, ms in rows:
		flag = ''
		if name.startswith('coreml-ane'):
			flag = '  PASS' if factor >= ANE_TARGET else '  BELOW %g×' % ANE_TARGET
		print('%-18s %8.0f×   %7.2f ms%s' % (name, factor, ms, flag))
	if ane and max(r[1] for r in ane) < ANE_TARGET:
		raise SystemExit('ANE stayed under %g× real time' % ANE_TARGET)
	return rows


def convert_coreai(args):
	export, ckpt, model, src = _build(args)  # fixed 1024-frame graph, same tensors the app feeds Core ML; fp32 in (app sends fp16)
	frames = args.frames or COREML_CHUNK_FRAMES
	mel, rmvpe = _pair(frames, 'zero')
	from coreai_torch import TorchConverter, get_decomp_table
	export = export.eval().cpu().float()
	example = (mel.detach().cpu().contiguous(), rmvpe.detach().cpu().contiguous())
	with torch.no_grad():
		ref = export(*example)
		ep = torch.export.export(export, args=example)
	ep = ep.run_decompositions(get_decomp_table())
	prog = TorchConverter().add_exported_program(ep, input_names=['mel', 'rmvpe'], output_names=['logits']).to_coreai()
	prog.optimize()
	out = args.output or os.path.join(DEFAULT_RUNS, 'YunaVAD.aimodel')
	from pathlib import Path
	dest = Path(out)
	dest.parent.mkdir(parents=True, exist_ok=True)
	if dest.exists():
		shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
	prog.save_asset(dest)
	_write_meta(out, ckpt, model, {'format': 'coreai', 'chunk_frames': frames, 'precision': 'fp32', 'checkpoint': src, 'ref_absmax': float(ref.abs().max())})
	print('saved', out, 'from', src, 'ref_absmax', float(ref.abs().max()), flush=True)
	return out


def main():
	p = argparse.ArgumentParser(description='Export LiquidVAD to Core ML, Core AI, or ONNX')
	p.add_argument('--to', required=True, choices=['coreml', 'coreai', 'onnx', 'all'], help='coreml and coreai are the Apple Neural Engine package. all runs both.')
	p.add_argument('--input', default=DEFAULT_CKPT)
	p.add_argument('--output', default=None)
	p.add_argument('--frames', type=int, default=None, help='Core ML chunk; ONNX trace length. Default 1024 / 512')
	p.add_argument('--fixed', action='store_true', help='ONNX only: disable dynamic time')
	p.add_argument('--random', action='store_true', help='export an untrained net, to prove the graph converts')
	p.add_argument('--bench', action='store_true', help='time CPU, GPU, and ANE')
	p.add_argument('--serve', action='store_true', help='ONNX only: serve web/ after export')
	p.add_argument('--port', type=int, default=8765)
	args = p.parse_args()
	onnx_path = ml_path = None
	export = None
	bench_dir = os.path.join(DEFAULT_RUNS, 'bench')
	if args.random and args.output is None:
		os.makedirs(bench_dir, exist_ok=True)
	if args.to in ('onnx', 'all'):
		if args.frames is None:
			args.frames = 512 if args.to == 'onnx' else COREML_CHUNK_FRAMES
		onnx_args = argparse.Namespace(**vars(args))
		if args.to == 'onnx':
			onnx_args.output = args.output or (os.path.join(bench_dir, 'liquid_vad.onnx') if args.random else None)
		else:
			onnx_args.output = os.path.join(bench_dir, 'liquid_vad.onnx') if args.random else None
		onnx_args.frames = args.frames if args.to == 'onnx' else 512
		onnx_path = convert_onnx(onnx_args)
		export = _build(args)[0]
	if args.to == 'coreai':
		if args.frames is None:
			args.frames = COREML_CHUNK_FRAMES
		convert_coreai(args)
		return
	if args.to in ('coreml', 'all'):
		if args.frames is None or args.to == 'all':
			args.frames = COREML_CHUNK_FRAMES
		ml_args = argparse.Namespace(**vars(args))
		if args.to == 'all':
			ml_args.output = os.path.join(bench_dir, 'liquid_vad.mlpackage') if args.random else None
		else:
			ml_args.output = args.output or (os.path.join(bench_dir, 'liquid_vad.mlpackage') if args.random else None)
		ml_path, _flex = convert_coreml(ml_args)
		export = _build(args)[0]
	if args.bench:
		bench(export, onnx_path, ml_path, COREML_CHUNK_FRAMES if ml_path else args.frames)


if __name__ == '__main__':
	main()
