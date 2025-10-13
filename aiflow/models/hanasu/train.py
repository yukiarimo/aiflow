import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from . import commons
from . import utils
from .mel_processing import mel_spectrogram_torch
from .models import (AVAILABLE_FLOW_TYPES,DurationDiscriminatorV2, MultiPeriodDiscriminator, SynthesizerTrn)
from .text import symbols, text_to_sequence
torch.backends.cudnn.benchmark = True
import random
import torch.utils.data
global_step = 0

class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """Loads audio, speaker_id, text pairs"""
    def __init__(self, audiopaths_sid_text, hparams):
        self.hparams = hparams
        self.audiopaths_sid_text = utils.load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.n_mel_channels = getattr(hparams, "n_mel_channels", 128)
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self.min_audio_len = getattr(hparams, "min_audio_len", 8192)

        random.seed(8)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """Filter text & store spec lengths."""
        audiopaths_sid_text_new, lengths = [], []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if not os.path.isfile(audiopath): continue
            if self.min_text_len <= len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                length = os.path.getsize(audiopath) // (2 * self.hop_length)
                if length < self.min_audio_len // self.hop_length: continue
                lengths.append(length)
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        print(len(self.lengths))

    def get_audio_text_speaker_pair(self, audiopath_sid_text, language='en'):
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text_norm = text_to_sequence(text, language)
        text = torch.LongTensor(text_norm)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return (text, spec, wav, sid)

    def get_audio(self, filename):
        audio, sampling_rate = utils.load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate: raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")
        audio_norm = (audio / self.max_wav_value).unsqueeze(0)

        segment_size = self.hparams.train.segment_size
        if audio_norm.size(1) < segment_size: audio_norm = torch.nn.functional.pad(audio_norm, (0, segment_size - audio_norm.size(1)), 'constant')

        spec_filename = filename.replace(".wav", ".mel.pt")
        if os.path.exists(spec_filename): spec = torch.load(spec_filename)
        else:
            spec = mel_spectrogram_torch(audio_norm, self.filter_length, self.n_mel_channels, self.sampling_rate, self.hop_length, self.win_length, self.hparams.mel_fmin, self.hparams.mel_fmax, center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_sid(self, sid): return torch.LongTensor([int(sid)])
    def __getitem__(self, index): return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])
    def __len__(self): return len(self.audiopaths_sid_text)

class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""
    def __init__(self, return_ids=False): self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len).zero_()
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len).zero_()
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len).zero_()

        for i, row_idx in enumerate(ids_sorted_decreasing):
            row = batch[row_idx]
            text, spec, wav = row[0], row[1], row[2]

            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            sid[i] = row[3]

        if self.return_ids: return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """Maintains similar input lengths in a batch."""
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i, length in enumerate(self.lengths):
            idx_bucket = self._bisect(length)
            if idx_bucket != -1: buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):
            if not buckets[i]:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for bucket in buckets:
            len_bucket = len(bucket)
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator(); g.manual_seed(self.epoch)

        if self.shuffle: indices = [torch.randperm(len(b), generator=g).tolist() for b in self.buckets]
        else: indices = [list(range(len(b))) for b in self.buckets]

        batches = []
        for i, bucket in enumerate(self.buckets):
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]
            rem = num_samples_bucket - len_bucket
            ids_bucket += ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)] # add extra samples
            ids_bucket = ids_bucket[self.rank::self.num_replicas] # subsample

            for j in range(len(ids_bucket) // self.batch_size): # batching
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None: hi = len(self.boundaries) - 1
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x <= self.boundaries[mid+1]: return mid
            elif x <= self.boundaries[mid]: return self._bisect(x, lo, mid)
            else: return self._bisect(x, mid + 1, hi)
        return -1

    def __len__(self): return self.num_samples // self.batch_size

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l

def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6060"

    hps = utils.get_hparams()
    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )

def run(rank, n_gpus, hps):
    net_dur_disc = None
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    posterior_channels = 128  # vits2

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    # some of these flags are not being used in the code and directly set in hps json file.
    # they are kept here for reference and prototyping.
    if (
        "use_transformer_flows" in hps.model.keys()
        and hps.model.use_transformer_flows == True
    ):
        use_transformer_flows = True
        transformer_flow_type = hps.model.transformer_flow_type
        print(f"Using transformer flows {transformer_flow_type} for VITS2")
        assert (
            transformer_flow_type in AVAILABLE_FLOW_TYPES
        ), f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
    else:
        print("Using normal flows for VITS1")
        use_transformer_flows = False

    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder == True
    ):
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
        use_spk_conditioned_encoder = True
    else:
        print("Using normal encoder for VITS1")
        use_spk_conditioned_encoder = False

    print("Using noise scaled MAS for VITS2")
    mas_noise_scale_initial = 0.01
    noise_scale_delta = 2e-6

    net_dur_disc = DurationDiscriminatorV2(
        hps.model.hidden_channels,
        hps.model.hidden_channels,
        3,
        0.1,
        gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
    ).cuda(rank)

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator().cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None

    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if net_dur_disc is not None:
        net_dur_disc = DDP(
            net_dur_disc, device_ids=[rank], find_unused_parameters=True)

    epoch_str = 1
    global_step = 0

    g_checkpoint_path = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
    if g_checkpoint_path is not None:
        try:
            utils.load_checkpoint(g_checkpoint_path, net_g, None)
            global_step = 0
            epoch_str = 1
            print(f"Loaded generator weights for fine-tuning: {g_checkpoint_path}")

            d_checkpoint_path = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
            if d_checkpoint_path is not None:
                utils.load_checkpoint(d_checkpoint_path, net_d, None)
                print(f"Loaded discriminator weights for fine-tuning: {d_checkpoint_path}")

            if net_dur_disc is not None:
                dur_checkpoint_path = utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth")
                if dur_checkpoint_path is not None:
                    utils.load_checkpoint(dur_checkpoint_path, net_dur_disc, None)
                    print(f"Loaded duration discriminator weights for fine-tuning: {dur_checkpoint_path}")
        except Exception as e:
            print(f"Could not load checkpoint, starting from scratch. Error: {e}")
            epoch_str = 1
            global_step = 0
    else:
        print("No generator checkpoint found, starting training from scratch")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc],
                [optim_g, optim_d, optim_dur_disc],
                [scheduler_g, scheduler_d, scheduler_dur_disc],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc],
                [optim_g, optim_d, optim_dur_disc],
                [scheduler_g, scheduler_d, scheduler_dur_disc],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()

def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()

    if rank == 0:
        loader = tqdm.tqdm(train_loader, desc="Loading train data")
    else:
        loader = train_loader
    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        speakers,
    ) in enumerate(loader):
        current_mas_noise_scale = (
            net_g.module.mas_noise_scale_initial
            - net_g.module.noise_scale_delta * global_step
        )
        net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
            rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        speakers = speakers.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
            ) = net_g(x, x_lengths, spec, spec_lengths, speakers)

            mel = spec

            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc

            # Duration Discriminator
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach()
                )
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                grad_norm_dur_disc = commons.clip_grad_value_(
                    net_dur_disc.parameters(), None
                )
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                if net_dur_disc is not None:
                    scalar_dict.update(
                        {
                            "loss/dur_disc/total": loss_dur_disc_all,
                            "grad_norm_dur_disc": grad_norm_dur_disc,
                        }
                    )
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )

                # if net_dur_disc is not None:
                #   scalar_dict.update({"loss/dur_disc_r" : f"{losses_dur_disc_r}"})
                #   scalar_dict.update({"loss/dur_disc_g" : f"{losses_dur_disc_g}"})
                #   scalar_dict.update({"loss/dur_gen" : f"{loss_dur_gen}"})

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )
                #if global_step > 0:
                #    utils.remove_old_checkpoints(hps.model_dir, prefixes=["G_*.pth", "D_*.pth", "DUR_*.pth"])
        global_step += 1

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))

def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = None, None, None, None, None, None, None
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
        ) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            speakers = speakers.cuda(0)

            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            speakers = speakers[:1]
            break

        # Check if we got any data from the eval_loader
        if x is None:
            print("Warning: eval_loader is empty, skipping evaluation")
            generator.train()
            return

        y_hat, attn, mask, *_ = generator.module.infer(
            x, x_lengths, speakers, max_len=1000
        )
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

        mel = spec

        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {"gen/audio": y_hat[0, :, : y_hat_lengths[0]]}
    if global_step == 0:
        image_dict.update(
            {"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
        )
        audio_dict.update({"gt/audio": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()

if __name__ == "__main__":
    main()
