# === SAE inference utilities that extend SAETrainer workflows ===
from __future__ import annotations
import os, json
from typing import Dict, List, Callable, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from tangermeme.predict import predict
from bpnetlite.bpnet import ControlWrapper

from .SAE_trainer import activations_to_rows

class MultiLayerActivationHook:
	"""Register forward hooks on multiple modules; stash outputs per layer."""
	def __init__(self, model: nn.Module, layer_names: List[str]):
		modules = dict(model.named_modules())
		missing = [ln for ln in layer_names if ln not in modules]
		if missing:
			raise ValueError(f"Layer names not found: {missing}")
		self.modules = modules
		self.layer_names = list(layer_names)
		self.store: Dict[str, Optional[torch.Tensor]] = {ln: None for ln in layer_names}
		self.handles: List[torch.utils.hooks.RemovableHandle] = []

	def _make_hook(self, name: str):
		def _hook(_, __, out):
			self.store[name] = out.detach()
		return _hook

	def register(self):
		if self.handles:
			return
		for ln in self.layer_names:
			h = self.modules[ln].register_forward_hook(self._make_hook(ln))
			self.handles.append(h)

	def clear(self):
		for ln in self.layer_names:
			self.store[ln] = None

	def remove(self):
		for h in self.handles:
			h.remove()
		self.handles.clear()


def save_metadata(layers, rows_per_locus, batch_files, out_dir, num_TopK):
	# Write a metadata dict to file
	meta = {
		"layers": layers,
		"rows_per_locus": rows_per_locus,
		"batch_files": batch_files,
		"out_dir": out_dir,
		"max_activations": num_TopK
	}

	with open(os.path.join(out_dir, "meta.json"), "w") as f:
		json.dump(meta, f, indent=2)
	
	print("Saved metadata")
	return meta


def prepare_activations(act, ch_max, center_len):
	# Permute the activations
	X_rows = activations_to_rows(act, center_len=center_len).float()
	
	# Normalize the rows by ch_max
	return X_rows / ch_max.unsqueeze(0)


def gather_topk_values(sae, X_rows):
	# Run samples through SAE
	_, z = sae(X_rows)

	# Grab the topk values and return a dict
	val, idx = torch.topk(z, k = sae.k, dim = 1, largest = True, sorted = False)

	return {'idx': idx.detach().cpu().numpy(), 'val': val.detach().cpu().numpy()}


@torch.no_grad()
def sae_topk_for_batch(acts, layers, sae_models, chmax_map, center_len, device) -> dict:
	# Create dictionary to store topk values
	per_layer_rows = {}

	# Loop through each layer and gather the TopK indices/values
	for ln in layers:

		# Get layer specific activations, ch_max, and sae_model
		act = acts[ln].to(device, non_blocking=True)
		ch_max = chmax_map[ln].to(device, non_blocking=True)
		sae = sae_models[ln]

		# Prepare activations for SAEs
		X_rows = prepare_activations(act, ch_max, center_len)

		# Gather TopK Values
		per_layer_rows[ln] = gather_topk_values(sae, X_rows)

		# Clear memory
		del act, X_rows
		if device.startswith("cuda") and torch.cuda.is_available():
			torch.cuda.empty_cache()

	return per_layer_rows


@torch.no_grad()
def collect_topk_indices_to_disk_from_trainer(trainer, sae_models, loader, out_dir) -> dict:
	# Create the output directory
	os.makedirs(out_dir, exist_ok=True)

	# Grab variables from trainer object
	model = ControlWrapper(trainer.model) # ControlWrapper is necessary for _predict(), but necessitates calling the model attribute in attaching hooks
	layers = trainer.layers
	device = trainer.device
	center_len = trainer.center_len
	chmax_map = trainer.chmax_map

	# Compute rows_per_locus - i.e. how many position vectors are we storing per locus
	first_batch = next(iter(loader))
	seq_length = first_batch[0].shape[-1]
	rows_per_locus = seq_length if center_len is None else center_len

	# Set up hooks
	hook = MultiLayerActivationHook(model.model, layers) # We want to hook the model and not the ControlWrapper, so we call model.model
	hook.register()

	# List to store batch file paths
	batch_files = []

	with tqdm(total=len(loader), desc="Collecting Top-K activations", unit="batch") as pbar:
		for bidx, batch in enumerate(loader, start=1):

			Xi, *_ = batch
			Xi = Xi.to(device, non_blocking=True)

			# Forward to populate hooks
			_ = predict(model, X=Xi, args=None, batch_size=Xi.shape[0], dtype="float32", device=device, verbose=False)

			# Collect per-layer top-k
			pl_rows = sae_topk_for_batch(
				acts=hook.store,
				layers=layers,
				sae_models=sae_models,
				chmax_map=chmax_map,
				center_len=center_len,
				device=device
			)
			hook.clear()

			# Combine indices and values into one array and save
			combined_arr = np.zeros((Xi.shape[0] * rows_per_locus, len(layers), sae_models[layers[0]].k , 2), dtype=np.float32)
			
			# Write the TopK indices and values to array
			for layer_idx, ln in enumerate(layers):
				idx_np = pl_rows[ln]["idx"]
				val_np = pl_rows[ln]["val"]
				combined_arr[:, layer_idx, :, 0] = idx_np.astype(np.float32)
				combined_arr[:, layer_idx, :, 1] = val_np

			# Save this batch and add the path to batch_files
			shard_path = os.path.join(out_dir, f"rows_batch_{bidx:05d}.npy")
			np.save(shard_path, combined_arr)
			batch_files.append(shard_path)

			# Clear memory
			del combined_arr, pl_rows
			if device.startswith("cuda") and torch.cuda.is_available():
				torch.cuda.empty_cache()
			
			# Update tqdm progress bar
			pbar.update(1)

	# Remove all hooks
	hook.remove()

	# Save metadata
	metadata = save_metadata(layers, rows_per_locus, batch_files, out_dir, sae_models[layers[0]].k)

	return metadata
