# models.py
import torch
import torch.nn as nn
from typing import List, Dict
import re
import os

# Return a parsed layer name
def safe_name(name: str) -> str:
	"""Return a filesystem-safe layer name."""
	return re.sub(r"[^A-Za-z0-9_\-]+", "_", name)

# Load models saved by SAETrainer.train_all()
def load_saes_from_dir(save_dir: str, layers: List[str], device: str) -> Dict[str, nn.Module]:
	"""
	Expects files named safe_name(layer) + '.pt' in save_dir.
	"""
	out: Dict[str, nn.Module] = {}
	for ln in layers:
		path = os.path.join(save_dir, f"{safe_name(ln)}.pt")
		if not os.path.exists(path):
			raise FileNotFoundError(f"Missing SAE for {ln}: {path}")
		sae = torch.load(path, map_location=device, weights_only = False)
		sae.eval().to(device)
		out[ln] = sae
	return out

# TopK Mask on activations
def _topk_mask(z: torch.Tensor, k: int) -> torch.Tensor:
	vals, idx = torch.topk(z, k, dim=1)
	mask = torch.zeros_like(z)
	mask.scatter_(1, idx, 1.0)
	return z * mask

# TopK SAE
class SAETopK(nn.Module):
	"""
	z = TopK(W_enc (x - b_pre))
	x_hat = W_dec z + b_dec
	"""
	def __init__(self, input_dim: int, latent_multiplier: float = 4.0, k_fraction: float = 0.05):
		super().__init__()

		# For the case where we're testing baseline (no sparsity)
		self.is_identity = (latent_multiplier == 1.0 and k_fraction == 1.0)
		if self.is_identity:
			self.input_dim = input_dim
			self.k = input_dim
			return

		self.input_dim = int(input_dim)
		self.latent_dim = int(self.input_dim * float(latent_multiplier))
		self.k = max(1, int(self.latent_dim * float(k_fraction)))

		self.b_pre = nn.Parameter(torch.zeros(self.input_dim))
		self.encoder = nn.Linear(self.input_dim, self.latent_dim, bias=False)
		self.decoder = nn.Linear(self.latent_dim, self.input_dim, bias=True)

		with torch.no_grad():
			self.decoder.weight.copy_(self.encoder.weight.T)

	def forward(self, x: torch.Tensor):
		if self.is_identity:
			return x, x

		pre = self.encoder(x - self.b_pre)
		z = _topk_mask(pre, self.k)
		xh = self.decoder(z)
		return xh, z
