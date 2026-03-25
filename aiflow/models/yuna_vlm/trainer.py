import json
import warnings
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map


class Dataset:
	def __init__(self, hf_dataset, config, processor, image_processor=None, take=None, split=None, image_resize_shape=None):
		if split is not None:
			self.dataset = hf_dataset[split]
		else:
			self.dataset = hf_dataset
		if take is not None:
			self.dataset = self.dataset.take(take)
		self.processor = processor
		self.config = config
		self.image_processor = image_processor
		self.image_resize_shape = image_resize_shape

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		from .utils import prepare_inputs

		item = self.dataset[idx]
		images = item.get("images", item.get("image", None))

		# It's assumed the prompt provides a pre-formatted string or ready message construct
		# since apply_chat_template was removed as requested.
		prompt = item.get("text", item.get("messages", item.get("conversations")))

		if images in (None, "", []):
			images = []

		prompts = [prompt]

		image_token_index = self.config["image_token_index"]

		inputs = prepare_inputs(processor=self.processor, images=images, audio=None, prompts=prompts, image_token_index=image_token_index, resize_shape=self.image_resize_shape)
		input_ids = inputs["input_ids"]
		pixel_values = inputs["pixel_values"]
		mask = inputs["attention_mask"]
		kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "pixel_values", "attention_mask"]}

		if mask is None:
			mask = mx.ones_like(input_ids)

		return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": mask, **kwargs}


def grad_checkpoint(layer):
	fn = type(layer).__call__

	def checkpointed_fn(model, *args, **kwargs):
		def inner_fn(params, *args, **kwargs):
			model.update(params)
			return fn(model, *args, **kwargs)

		return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

	type(layer).__call__ = checkpointed_fn


class TrainingArgs:
	def __init__(self, batch_size=4, iters=100, val_batches=25, steps_per_report=10, steps_per_eval=200, steps_per_save=100, max_seq_length=2048, adapter_file="adapters.safetensors", grad_checkpoint=False):
		self.batch_size = batch_size
		self.iters = iters
		self.val_batches = val_batches
		self.steps_per_report = steps_per_report
		self.steps_per_eval = steps_per_eval
		self.steps_per_save = steps_per_save
		self.max_seq_length = max_seq_length
		self.adapter_file = adapter_file
		self.grad_checkpoint = grad_checkpoint


def default_loss(model, inputs, targets, lengths):
	logits = model(inputs)
	logits = logits.astype(mx.float32)

	length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

	ce = nn.losses.cross_entropy(logits, targets) * length_mask
	ntoks = length_mask.sum()
	ce = ce.sum() / ntoks

	return ce, ntoks


class Trainer:
	def __init__(self, model, optimizer, train_on_completions=False, assistant_id=77091, clip_gradients=None):
		self.model = model
		self.optimizer = optimizer
		self.train_on_completions = train_on_completions
		self.assistant_id = assistant_id
		self.clip_gradients = clip_gradients

	def loss_fn(self, model, batch):
		pixel_values = batch["pixel_values"]
		input_ids = batch["input_ids"]
		attention_mask = batch["attention_mask"]
		lengths = mx.sum(attention_mask, axis=1)
		labels = input_ids[:, 1:]

		batch_size, seq_length = input_ids.shape

		if self.train_on_completions:
			weight_mask = mx.ones_like(attention_mask)
			assistant_response_index = np.where(input_ids == self.assistant_id)[1]
			range_matrix = mx.repeat(mx.expand_dims(mx.arange(seq_length), 0), batch_size, axis=0)
			assistant_mask = range_matrix <= mx.array(assistant_response_index).reshape(-1, 1)
			weight_mask = mx.where(assistant_mask, mx.zeros_like(weight_mask), weight_mask)[:, 1:]
		else:
			weight_mask = None

		input_ids = input_ids[:, :-1]
		kwargs = {k: v for k, v in batch.items() if k not in ["input_ids", "pixel_values", "attention_mask"]}

		outputs = model(input_ids, pixel_values, attention_mask, **kwargs)
		logits = outputs.logits.astype(mx.float32)

		def align_logits_with_labels(logits, labels):
			if logits.shape[1] < labels.shape[1]:
				pad_length = labels.shape[1] - logits.shape[1]
				pad_width = ((0, 0), (0, pad_length), (0, 0))
				return mx.pad(logits, pad_width, mode="constant", constant_values=-100)
			elif logits.shape[1] > labels.shape[1]:
				return logits[:, -labels.shape[1]:, :]
			return logits

		logits = align_logits_with_labels(logits, labels)
		length_mask = mx.arange(input_ids.shape[1])[None, :] < lengths[:, None]

		ce = (nn.losses.cross_entropy(logits, labels, weights=weight_mask) * length_mask)
		ntoks = length_mask.sum()
		ce = ce.sum() / ntoks

		return ce

	def train_step(self, batch):
		loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
		loss, grads = loss_and_grad_fn(self.model, batch)

		if self.clip_gradients is not None:
			grads = tree_map(lambda g: mx.clip(g, -self.clip_gradients, self.clip_gradients), grads)

		self.optimizer.update(self.model, grads)
		return loss

	@mx.compile
	def train_epoch(self, dataloader):
		total_loss = 0
		for batch in dataloader:
			loss = self.train_step(batch)
			mx.eval(self.model, self.optimizer.state)
			total_loss += loss
		return total_loss / len(dataloader)


def save_adapter(model, adapter_file):
	path = Path(adapter_file)
	if hasattr(model.config, "lora"):
		with open(path.parent / "adapter_config.json", "w") as f:
			json.dump(model.config.lora, f)
	flattened_tree = tree_flatten(model.trainable_parameters())
	mx.save_safetensors(str(adapter_file), dict(flattened_tree))
