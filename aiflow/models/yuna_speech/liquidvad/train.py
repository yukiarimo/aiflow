import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import LiquidVADSet, collate_vad
from models import LiquidVAD, load_vad, save_checkpoint, vad_loss, voiced_pitch_loss
from utils import DEFAULT_CKPT, DEFAULT_DATA, DEFAULT_RUNS, HOP_LENGTH, SAMPLE_RATE, count_params, decode_segments, ensure_caches, get_device, load_npz, read_json, takes_dir, write_json

PITCH_W = 0.25
LOG_KEYS = ('loss', 'speech', 'bce', 'qdr', 'start', 'stop', 'stop_bce', 'late', 'pitch', 'acc', 'clean', 'noisy', 'pitch_on', 'pitch_off', 'stop_at', 'stop_on_end', 'stop_on_pause', 'grad', 'gate', )


def _masked_acc(pred, speech, mask):
	m = mask > 0.5
	ok = ((pred[..., 0] >= 0.5) == (speech >= 0.5))
	return (ok & m).float().sum() / m.float().sum().clamp_min(1)


def _metrics(logits, batch):
	mask = batch['mask'] > 0.5
	pred = torch.sigmoid(logits)
	acc = _masked_acc(pred, batch['speech'], batch['mask'])
	stop_w = batch['stop'] * mask.float()
	stop_at = (pred[..., 2] * stop_w).sum() / stop_w.sum().clamp_min(1e-6)
	clean = noisy = acc
	if pred.shape[0] >= 2 and pred.shape[0] % 2 == 0:
		clean = _masked_acc(pred[0::2], batch['speech'][0::2], batch['mask'][0::2])
		noisy = _masked_acc(pred[1::2], batch['speech'][1::2], batch['mask'][1::2])
	return acc.item(), stop_at.item(), clean.item(), noisy.item()


def drop_pitch(rmvpe):
	"""Half the rows lose pitch entirely. The rest lose random frames. Targets stay intact."""
	out = rmvpe.clone()
	kill = torch.rand(out.shape[0], device=out.device) < 0.5
	out[kill] = 0
	holes = torch.rand(out.shape[0], 1, out.shape[-1], device=out.device) < 0.35
	out = out.masked_fill(holes & ~kill[:, None, None], 0)
	return out, kill


def _blank():
	return {k: 0.0 for k in LOG_KEYS}


def run_epoch(model, loader, device, optimizer=None):
	train = optimizer is not None
	model.train(train)
	totals = _blank()
	n = 0
	preview = None
	for batch in loader:
		batch = {k: v.to(device, non_blocking=False) for k, v in batch.items()}
		rmvpe = batch['rmvpe']
		if train:
			optimizer.zero_grad(set_to_none=True)
			seen, killed = drop_pitch(rmvpe)
		else:
			seen, killed = rmvpe, torch.zeros(rmvpe.shape[0], dtype=torch.bool, device=device)
		logits, pitch = model(batch['mel'], seen, aux=True)
		loss, parts = vad_loss(logits, batch['speech'], batch['start'], batch['stop'], batch['mask'])
		pl = voiced_pitch_loss(pitch, rmvpe, batch['mask'])
		loss = loss + PITCH_W * pl
		parts['pitch'] = pl.item()
		grad = 0.0
		if train:
			loss.backward()
			grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0))
			optimizer.step()
		acc, stop_at, clean, noisy = _metrics(logits.detach(), batch)
		training = model.training
		model.eval()
		with torch.no_grad():
			z_logits = model(batch['mel'], torch.zeros_like(rmvpe), aux=False)
			off = _masked_acc(torch.sigmoid(z_logits), batch['speech'], batch['mask'])
			on_m = ~killed
			if on_m.any():
				on = _masked_acc(torch.sigmoid(logits.detach()[on_m]), batch['speech'][on_m], batch['mask'][on_m])
			else:
				on = acc
		if training:
			model.train()
		gates = model.gate_stats()
		totals['loss'] += loss.item()
		totals['speech'] += parts['speech']
		totals['bce'] += parts['bce']
		totals['qdr'] += parts['qdr']
		totals['start'] += parts['start']
		totals['stop'] += parts['stop']
		totals['stop_bce'] += parts['stop_bce']
		totals['late'] += parts['late']
		totals['pitch'] += parts['pitch']
		totals['acc'] += acc
		totals['clean'] += clean
		totals['noisy'] += noisy
		totals['pitch_on'] += float(on)
		totals['pitch_off'] += float(off)
		totals['stop_at'] += stop_at
		totals['stop_on_end'] += parts['stop_on_end']
		totals['stop_on_pause'] += parts['stop_on_pause']
		totals['grad'] += grad
		totals['gate'] += float(gates.mean()) if gates is not None else 0.0
		if preview is None:
			preview = {'logits': logits.detach().float().cpu(), 'speech': batch['speech'].detach().cpu(), 'start': batch['start'].detach().cpu(), 'stop': batch['stop'].detach().cpu(), 'mask': batch['mask'].detach().cpu(), 'gates': gates, }
		n += 1
	if not n:
		return totals, None
	return {k: v / n for k, v in totals.items()}, preview


def _log_preview(writer, tag, preview, step):
	if preview is None:
		return
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	n = int(preview['mask'][0].sum().item())
	n = max(n, 1)
	prob = torch.sigmoid(preview['logits'][0, :n]).numpy()
	t = np.arange(n)
	fig, ax = plt.subplots(3, 1, figsize=(12, 6.2), sharex=True)
	pairs = (('speech', preview['speech'][0, :n].numpy(), prob[:, 0], '#7aa2ff'), ('start', preview['start'][0, :n].numpy(), prob[:, 1], '#5ad68a'), ('stop', preview['stop'][0, :n].numpy(), prob[:, 2], '#ff6b9d'), )
	for a, (name, lab, pr, color) in zip(ax, pairs):
		a.fill_between(t, lab, color='#2a2a36', alpha=0.9, label='label')
		a.plot(t, pr, color=color, lw=1.2, label=name)
		a.set_ylim(-0.05, 1.05)
		a.set_ylabel(name)
		a.legend(loc='upper right', fontsize=8)
	ax[-1].set_xlabel('frame (10 ms)')
	fig.tight_layout()
	writer.add_figure(tag + '/curves', fig, step)
	plt.close(fig)
	writer.add_histogram(tag + '/speech_prob', prob[:, 0], step)
	writer.add_histogram(tag + '/start_prob', prob[:, 1], step)
	writer.add_histogram(tag + '/stop_prob', prob[:, 2], step)
	if preview['gates'] is not None:
		writer.add_histogram(tag + '/liquid_gate', preview['gates'].numpy(), step)


def checkpoint_dir(out_path):
	return os.path.join(os.path.dirname(os.path.abspath(out_path)), 'ckpts')


def _iou(a, b):
	inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
	union = (a[1] - a[0]) + (b[1] - b[0]) - inter
	return inter / union if union > 1e-6 else 0.0


def _match_segments(ref, hyp, min_iou=0.30):
	used = set()
	pairs = []
	for r in ref:
		best, bi = 0.0, None
		for i, h in enumerate(hyp):
			if i in used:
				continue
			score = _iou(r, h)
			if score > best:
				best, bi = score, i
		if bi is not None and best >= min_iou:
			used.add(bi)
			pairs.append((r, hyp[bi]))
	return pairs, len(ref) - len(pairs), len(hyp) - len(pairs)


@torch.no_grad()
def _probs(model, mel, rmvpe, device):
	model.eval()
	logits = model(torch.from_numpy(mel).unsqueeze(0).to(device), torch.from_numpy(rmvpe).unsqueeze(0).to(device), aux=False)
	return torch.sigmoid(logits[0]).float().cpu().numpy()


def evaluate_takes(model, pairs, device):
	"""Score decoded cuts against the click labels. pairs are (cache npz, take json)."""
	rows = []
	hit = miss = extra = 0
	start_err = []
	stop_err = []
	hit_off = 0
	for cache_path, json_path in pairs:
		z = load_npz(cache_path)
		meta = read_json(json_path) if os.path.isfile(json_path) else {'utterances': []}
		ref = [(float(u['start']), float(u['stop'])) for u in meta.get('utterances', []) if 'start' in u and 'stop' in u]
		prob = _probs(model, z['mel'], z['rmvpe'], device)
		hyp = decode_segments(prob[:, 0], prob[:, 1], prob[:, 2], hop=HOP_LENGTH, sr=SAMPLE_RATE)
		zeros = np.zeros_like(z['rmvpe'])
		prob_off = _probs(model, z['mel'], zeros, device)
		hyp_off = decode_segments(prob_off[:, 0], prob_off[:, 1], prob_off[:, 2], hop=HOP_LENGTH, sr=SAMPLE_RATE)
		matched, missed, extras = _match_segments(ref, hyp)
		matched_off, _, _ = _match_segments(ref, hyp_off)
		hit += len(matched)
		miss += missed
		extra += extras
		hit_off += len(matched_off)
		for r, h in matched:
			start_err.append(abs(h[0] - r[0]))
			stop_err.append(abs(h[1] - r[1]))
		rows.append({'take': os.path.splitext(os.path.basename(cache_path))[0], 'ref': len(ref), 'hit': len(matched), 'miss': missed, 'extra': extras, 'hit_pitch_off': len(matched_off), 'start_ms': 1000.0 * float(np.mean(start_err[-len(matched):])) if matched else None, 'stop_ms': 1000.0 * float(np.mean(stop_err[-len(matched):])) if matched else None, })
	nref = hit + miss
	summary = {'takes': len(pairs), 'ref': nref, 'hit': hit, 'miss': miss, 'extra': extra, 'hit_pitch_off': hit_off, 'hit_rate': hit / max(1, nref), 'hit_rate_pitch_off': hit_off / max(1, nref), 'start_ms': 1000.0 * float(np.mean(start_err)) if start_err else None, 'stop_ms': 1000.0 * float(np.mean(stop_err)) if stop_err else None, 'rows': rows, }
	return summary


def _print_eval(tag, summary):
	start = '—' if summary['start_ms'] is None else '%.0f ms' % summary['start_ms']
	stop = '—' if summary['stop_ms'] is None else '%.0f ms' % summary['stop_ms']
	print('eval %-8s  clips %d/%d hit  pitch-off %d/%d  extra %d  start %s  stop %s' % (tag, summary['hit'], summary['ref'], summary['hit_pitch_off'], summary['ref'], summary['extra'], start, stop))
	for row in summary['rows']:
		print('  %-10s  %d/%d hit  pitch-off %d  extra %d' % (row['take'], row['hit'], row['ref'], row['hit_pitch_off'], row['extra']))


def _save_tree(model, epoch, score, out_path, folder, summary, every, rows):
	os.makedirs(folder, exist_ok=True)
	metrics = {k: summary[k] for k in ('hit', 'miss', 'extra', 'hit_pitch_off', 'hit_rate', 'hit_rate_pitch_off', 'start_ms', 'stop_ms', 'ref') if summary}
	last_path = os.path.join(folder, 'last.pt')
	save_checkpoint(last_path, model, epoch, score, extra={'tag': 'last', 'metrics': metrics})
	epoch_path = None
	if every > 0 and (epoch % every == 0 or epoch == 1):
		epoch_path = os.path.join(folder, 'epoch_%04d.pt' % epoch)
		save_checkpoint(epoch_path, model, epoch, score, extra={'tag': 'epoch_%04d' % epoch, 'metrics': metrics})
		rows.append({'epoch': epoch, 'loss': score, 'path': epoch_path, **metrics})
		write_json(os.path.join(folder, 'index.json'), rows)
	return last_path, epoch_path


def main():
	p = argparse.ArgumentParser(description='Train LiquidVAD on click-labeled takes')
	p.add_argument('--data', default=DEFAULT_DATA)
	p.add_argument('--out', default=DEFAULT_CKPT)
	p.add_argument('--rmvpe', default=None)
	p.add_argument('--device', default='mps')
	p.add_argument('--epochs', type=int, default=80)
	p.add_argument('--batch', type=int, default=4, help='takes per step; each is stacked clean and noisy')
	p.add_argument('--lr', type=float, default=2e-3)
	p.add_argument('--hidden', type=int, default=64)
	p.add_argument('--crop', type=float, default=6.0)
	p.add_argument('--val', type=int, default=2, help='hold out last N takes')
	p.add_argument('--save-every', type=int, default=5, help='write runs/ckpts/epoch_XXXX.pt this often, plus last.pt every epoch')
	p.add_argument('--eval-every', type=int, default=5, help='decode the held-out takes and score them against the clicks')
	p.add_argument('--eval', default=None, help='score this checkpoint and exit (a .pt path)')
	p.add_argument('--split', default='val', choices=['val', 'train', 'all'], help='which takes --eval reads')
	p.add_argument('--cache-only', action='store_true')
	p.add_argument('--force-cache', action='store_true')
	p.add_argument('--logdir', default=os.path.join(DEFAULT_RUNS, 'tb'), help='TensorBoard log dir')
	args = p.parse_args()

	device = get_device(args.device)
	print('device:', device)
	from utils import list_take_jsons, read_json, wav_is_silent
	dead = []
	for jp in list_take_jsons(args.data):
		meta = read_json(jp)
		wav = meta.get('wav') or os.path.join(os.path.dirname(jp), os.path.basename(meta.get('path', '')))
		if not os.path.isfile(wav):
			wav = os.path.join(os.path.dirname(jp), os.path.basename(meta.get('path', '')))
		if wav_is_silent(wav):
			dead.append(os.path.basename(wav))
	if dead:
		raise SystemExit('silent takes (all zeros): %s\nre-record — System Settings → Privacy → Microphone → enable Cursor/Terminal, then:\n  python dataset.py --start-index 1' % ', '.join(dead))
	caches = ensure_caches(args.data, rmvpe_path=args.rmvpe, device=device, force=args.force_cache)
	if not caches:
		raise SystemExit('no takes in %s — run dataset.py first' % args.data)
	if args.cache_only:
		print('cached', len(caches), 'takes')
		return

	n_val = min(max(0, args.val), max(0, len(caches) - 1))
	val_p = caches[-n_val:] if n_val else []
	train_p = caches[:-n_val] if n_val else list(caches)
	if not train_p:
		train_p, val_p = caches[:-1], caches[-1:]

	def _pairs(paths):
		out = []
		for cp in paths:
			stem = os.path.splitext(os.path.basename(cp))[0]
			out.append((cp, os.path.join(takes_dir(args.data), stem + '.json')))
		return out

	if args.eval:
		if not os.path.isfile(args.eval):
			raise SystemExit('no checkpoint at %s' % args.eval)
		model, ckpt = load_vad(args.eval, device=device)
		chosen = {'val': val_p, 'train': train_p, 'all': caches}[args.split]
		summary = evaluate_takes(model, _pairs(chosen), device)
		print('checkpoint', args.eval, 'epoch', ckpt.get('epoch'), 'split', args.split)
		_print_eval(args.split, summary)
		return

	train_set = LiquidVADSet(train_p, augment=True, crop_sec=args.crop)
	val_set = LiquidVADSet(val_p, augment=False, crop_sec=0.0) if val_p else None
	train_loader = DataLoader(train_set, batch_size=min(args.batch, len(train_set)), shuffle=True, drop_last=False, collate_fn=collate_vad, num_workers=0)
	val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_vad, num_workers=0) if val_set else None

	model = LiquidVAD(hidden=args.hidden).to(device)
	opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
	sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
	os.makedirs(args.logdir, exist_ok=True)
	writer = SummaryWriter(args.logdir)
	print('params:', f'{count_params(model):,}', 'train takes:', len(train_set), 'val takes:', 0 if val_set is None else len(val_set))
	print('pitch input is dropped on half the rows and on random frames; voiced f0 is the only pitch grade')
	print('tensorboard:', args.logdir)
	print('  tensorboard --logdir', args.logdir)

	best, best_path = 1e9, args.out
	folder = checkpoint_dir(args.out)
	history = []
	print('checkpoints:', folder, 'every', args.save_every, 'epochs, plus last.pt each epoch and', args.out, 'when val improves')
	t0 = time.perf_counter()
	for epoch in range(1, args.epochs + 1):
		tr, preview = run_epoch(model, train_loader, device, opt)
		msg = 'epoch %3d  loss %.4f  bce %.4f  qdr %.4f  start %.4f  stop %.4f  late %.4f  pitch %.4f  acc %.3f  clean %.3f  noisy %.3f  pitch-on %.3f  pitch-off %.3f  end %.3f  pause %.3f' % (epoch, tr['loss'], tr['bce'], tr['qdr'], tr['start'], tr['stop'], tr['late'], tr['pitch'], tr['acc'], tr['clean'], tr['noisy'], tr['pitch_on'], tr['pitch_off'], tr['stop_on_end'], tr['stop_on_pause'])
		score = tr['loss']
		for key in LOG_KEYS:
			writer.add_scalar('train/' + key, tr[key], epoch)
		writer.add_scalars('train/stop_where', {'on_click': tr['stop_on_end'], 'on_pause': tr['stop_on_pause']}, epoch)
		writer.add_scalars('train/pitch_input', {'present': tr['pitch_on'], 'zeroed': tr['pitch_off']}, epoch)
		writer.add_scalar('lr', opt.param_groups[0]['lr'], epoch)
		_log_preview(writer, 'train', preview, epoch)
		if val_loader:
			va, vpreview = run_epoch(model, val_loader, device, None)
			msg += '  | val %.4f acc %.3f pitch-off %.3f end %.3f pause %.3f' % (va['loss'], va['acc'], va['pitch_off'], va['stop_on_end'], va['stop_on_pause'])
			score = va['loss']
			for key in LOG_KEYS:
				writer.add_scalar('val/' + key, va[key], epoch)
			writer.add_scalars('val/stop_where', {'on_click': va['stop_on_end'], 'on_pause': va['stop_on_pause']}, epoch)
			writer.add_scalars('val/pitch_input', {'present': va['pitch_on'], 'zeroed': va['pitch_off']}, epoch)
			_log_preview(writer, 'val', vpreview, epoch)
		summary = None
		if val_p and (epoch == 1 or epoch % max(1, args.eval_every) == 0 or epoch == args.epochs):
			summary = evaluate_takes(model, _pairs(val_p), device)
			_print_eval('val', summary)
			writer.add_scalar('val/hit_rate', summary['hit_rate'], epoch)
			writer.add_scalar('val/hit_rate_pitch_off', summary['hit_rate_pitch_off'], epoch)
			writer.add_scalar('val/extra', summary['extra'], epoch)
			if summary['start_ms'] is not None:
				writer.add_scalar('val/start_ms', summary['start_ms'], epoch)
			if summary['stop_ms'] is not None:
				writer.add_scalar('val/stop_ms', summary['stop_ms'], epoch)
		print(msg, flush=True)
		sched.step()
		_save_tree(model, epoch, score, args.out, folder, summary or {}, args.save_every, history)
		if score <= best:
			best = score
			metrics = {}
			if summary:
				metrics = {k: summary[k] for k in ('hit_rate', 'hit_rate_pitch_off', 'start_ms', 'stop_ms', 'extra', 'ref')}
			save_checkpoint(best_path, model, epoch, score, extra={'tag': 'best', 'metrics': metrics})
			save_checkpoint(os.path.join(folder, 'best.pt'), model, epoch, score, extra={'tag': 'best', 'metrics': metrics})
			writer.add_scalar('best/loss', best, epoch)
		writer.flush()
	writer.close()
	print('saved', best_path, 'best %.4f' % best, 'in %.1fs' % (time.perf_counter() - t0))
	print('other epochs:', folder)
	print('tensorboard --logdir', args.logdir)


if __name__ == '__main__':
	main()
