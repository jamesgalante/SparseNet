# sae_trainer.py
from __future__ import annotations
import os, csv, time, re
from typing import Optional, Type

import torch
import torch.nn.functional as F

from .models import safe_name

@torch.no_grad()
def capture_activations(model, layer_name: str, Xi: torch.Tensor, Xi_ctl: torch.Tensor, device: str) -> torch.Tensor:
	"""Register hook, run prediction, remove, and return activations [B,C,L]."""
	store = {}

	def hook(_, __, out):
		store["act"] = out.detach()

	handle = dict(model.named_modules())[layer_name].register_forward_hook(hook)

	from tangermeme.predict import predict
	_ = predict(
		model,
		X=Xi,
		args=[Xi_ctl],
		batch_size=Xi.shape[0],
		dtype="float32",
		device=device,
		verbose=False,
	)

	handle.remove()
	return store["act"]


def activations_to_rows(act: torch.Tensor, center_len: Optional[int]) -> torch.Tensor:
	"""[B,C,L] → optional center crop → [B,L,C] → [N,C]."""
	B, C, L = act.shape
	if (center_len is None) or (center_len >= L):
		act_BCL = act
	else:
		s = (L - center_len) // 2
		e = s + center_len
		act_BCL = act[:, :, s:e]
	return act_BCL.permute(0, 2, 1).reshape(-1, C)


@torch.no_grad()
def compute_ch_max(model, layer_name: str, train_loader, device: str) -> torch.Tensor:
	"""Compute max absolute activation per channel for normalization."""
	ch_max = None
	for (Xi, Xi_ctl, *_) in train_loader:
		Xi, Xi_ctl = Xi.to(device, non_blocking=True), Xi_ctl.to(device, non_blocking=True)
		act = capture_activations(model, layer_name, Xi, Xi_ctl, device)  # [B,C,L]
		m = act.abs().amax(dim=(0, 2))	# [C]
		ch_max = m if ch_max is None else torch.maximum(ch_max, m)
	return ch_max.clamp_min(1e-6)


# === Main trainer class ===

class SAETrainer:
	"""
	Minimal trainer:
	  1) Load BPNet model
	  2) Find ReLU layers
	  3) Train SAEs for specified layers
	"""

	def __init__(self, model_path: str, device: Optional[str] = None, center_len: Optional[int] = 1000):
		import bpnetlite

		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.center_len = center_len

		# Load BPNet model
		self.model = bpnetlite.bpnet.BasePairNet.from_bpnet(model_path).to(self.device).eval()

		# Collect ReLU layers
		self.layers = [
			name
			for name, module in self.model.named_modules()
			if len(list(module.children())) == 0 and "relu" in name
		]

		# Initialize a dictionary to store the channel max for each layer
		self.chmax_map = {}

	def train_layer(
		self,
		*,
		layer_name: str,
		train_loader,
		sae_cls: Type[torch.nn.Module],
		sae_kwargs: dict,
		epochs: int = 3,
		inner_bs: int = 16384,
		lr: float = 1e-3,
		log_every: int = 250,
		csv_log_path: Optional[str] = None,
		save_path: Optional[str] = None,
	) -> None:
		"""Train one SAE for a given layer."""
		print(f"[{layer_name}] Computing channel max for normalization...")
		ch_max = compute_ch_max(self.model, layer_name, train_loader, self.device)
		self.chmax_map[layer_name] = ch_max.detach().cpu()
		input_dim = int(ch_max.numel())

		sae = sae_cls(input_dim=input_dim, **sae_kwargs).to(self.device).train()
		params = list(sae.parameters())
		opt = torch.optim.Adam(params, lr=lr) if params else None

		writer = None
		if csv_log_path:
			os.makedirs(os.path.dirname(csv_log_path) or ".", exist_ok=True)
			fp = open(csv_log_path, "w", newline="")
			writer = csv.writer(fp)
			writer.writerow(["epoch", "batch_idx", "last_loss", "avg_batch_time_s", "cuda_mem_MB", "cuda_peak_MB"])
			fp.flush()

		use_cuda_stats = str(self.device).startswith("cuda") and torch.cuda.is_available()

		print(f"[{layer_name}] Training for {epochs} epochs...")
		for ep in range(1, epochs + 1):
			batch_times = []
			if use_cuda_stats:
				torch.cuda.reset_peak_memory_stats(self.device)

			for bidx, (Xi, Xi_ctl, *_) in enumerate(train_loader, start=1):
				t0 = time.time()
				Xi, Xi_ctl = Xi.to(self.device, non_blocking=True), Xi_ctl.to(self.device, non_blocking=True)

				with torch.no_grad():
					act = capture_activations(self.model, layer_name, Xi, Xi_ctl, self.device)
					X = activations_to_rows(act, self.center_len).float()
					X = X / ch_max.unsqueeze(0)

				last_loss = float("nan")
				N = X.shape[0]
				for i in range(0, N, inner_bs):
					xb = X[i : i + inner_bs]
					xh, _ = sae(xb)
					loss = F.mse_loss(xh, xb)
					if opt is not None:
						opt.zero_grad(set_to_none=True)
						loss.backward()
						opt.step()
					last_loss = float(loss.item())

				bt = time.time() - t0
				batch_times.append(bt)
				avg_bt = sum(batch_times) / len(batch_times)
				mem = peak = 0.0
				if use_cuda_stats:
					mem = torch.cuda.memory_allocated(self.device) / (1024**2)
					peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)

				if writer and (bidx % log_every) == 0:
					writer.writerow([ep, bidx, last_loss, avg_bt, f"{mem:.0f}", f"{peak:.0f}"])
					fp.flush()

			print(f"[{layer_name}] Epoch {ep} finished.")

		if writer:
			fp.close()

		if save_path:
			os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
			torch.save(sae, save_path)

	def train_all(
		self,
		*,
		train_loader,
		sae_cls: Type[torch.nn.Module],
		sae_kwargs: dict,
		save_dir: str = "sae_full_models",
		logs_dir: str = "training_logs",
		epochs: int = 3,
		inner_bs: int = 16384,
		lr: float = 1e-3,
		log_every: int = 250,
	) -> None:
		"""Train SAEs for all discovered layers."""
		os.makedirs(save_dir, exist_ok=True)
		os.makedirs(logs_dir, exist_ok=True)

		total = len(self.layers)
		for i, ln in enumerate(self.layers, 1):
			print(f"\n[{i}/{total}] Training layer: {ln}")
			csv_log = os.path.join(logs_dir, f"{safe_name(ln)}.csv")
			out_path = os.path.join(save_dir, f"{safe_name(ln)}.pt")

			self.train_layer(
				layer_name=ln,
				train_loader=train_loader,
				sae_cls=sae_cls,
				sae_kwargs=sae_kwargs,
				epochs=epochs,
				inner_bs=inner_bs,
				lr=lr,
				log_every=log_every,
				csv_log_path=csv_log,
				save_path=out_path,
			)
			print(f"  ✓ Saved: {out_path}")
