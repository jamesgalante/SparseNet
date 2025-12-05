import numpy as np
import heapq
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from tangermeme.plot import plot_logo


def get_receptive_field_bounds(row_idx, rows_per_locus, receptive_field, seq_length):
	# Map row index to receptive field position in input sequence
	position_in_locus = row_idx % rows_per_locus
	center_offset = (seq_length - rows_per_locus) // 2
	center_pos = center_offset + position_in_locus
	rf_half = receptive_field // 2
	return center_pos - rf_half, center_pos - rf_half + receptive_field


def create_PWM_for_top_N_nodes(layer, receptive_field, num_top_nodes, data_dir, out_dir, loader, latent_dim, num_samples_per_node=1000):
	"""Create PWMs for top N most frequent nodes in a layer."""
	
	# Load metadata ===
	with open(os.path.join(data_dir, "meta.json")) as f:
		meta = json.load(f)
	batch_files = meta["batch_files"]
	rows_per_locus = meta["rows_per_locus"]
	num_layers = len(meta["layers"])
	
	# Get sequence length from loader
	Xi, *_ = next(iter(loader))
	seq_length = Xi.shape[-1]
	
	# Count node activations
	counts = np.zeros(latent_dim, dtype=np.int64)
	for path in tqdm(batch_files, desc="Counting activations"):
		arr = np.load(path, mmap_mode="r")
		idx = arr[:, layer, :, 0].astype(np.int32, copy=False)
		counts += np.bincount(idx.ravel(), minlength=latent_dim)
	
	# Rank nodes by frequency
	node_order = np.argsort(counts)[::-1]
	
	# Collect top K activations for each top node
	top_activations = [[] for _ in range(num_top_nodes)]
	
	# Go through each batch and store the top activating (batch idx, row) for each of the top nodes
	for batch_idx, path in enumerate(tqdm(batch_files, desc="Collecting top activations"), start=1):
		# Grab the batch information
		arr = np.load(path, mmap_mode="r")
		idx = arr[:, layer, :, 0].astype(np.int32, copy=False)
		val = np.abs(arr[:, layer, :, 1])

		# For each top seen node, store the top activating (batch, row) information
		for rank in range(num_top_nodes):

			# Create a mask to isolate values for the current top seen node
			node_idx = int(node_order[rank])
			mask = (idx == node_idx)
			if not mask.any():
				continue

			# Isolate values for the current top seen node
			rows, _ = np.where(mask)
			mags = val[mask]

			# Grab the running top activations for the current top seen node
			heap = top_activations[rank]

			# For each index activation, add row and batch idx to heap if activation is greater than current minimum 
			for mag, r in zip(mags, rows):
				item = (float(mag), batch_idx, int(r))
				if len(heap) < num_samples_per_node:
					heapq.heappush(heap, item)
				else:
					heapq.heappushpop(heap, item)
	
	# Sort each top node's highest activating samples by magnitude (descending)
	for rank in range(num_top_nodes):
		top_activations[rank] = sorted(top_activations[rank], reverse=True)
	
	# Build batch lookup for efficient sequence extraction
	batch_lookup = {}
	for rank, activations in enumerate(top_activations):
		for mag, batch_idx, row in activations:
			if batch_idx not in batch_lookup:
				batch_lookup[batch_idx] = {}
			if rank not in batch_lookup[batch_idx]:
				batch_lookup[batch_idx][rank] = []
			batch_lookup[batch_idx][rank].append(row)
	
	# Extract sequences in a single pass
	node_sequences = {rank: [] for rank in range(num_top_nodes)}
	
	for batch_idx, batch in enumerate(tqdm(loader, desc="Extracting sequences"), start=1):
		if batch_idx not in batch_lookup:
			continue
		
		Xi, *_ = batch
		Xi_np = Xi.cpu().numpy()
		
		for rank, rows in batch_lookup[batch_idx].items():
			for row in rows:
				locus_idx = row // rows_per_locus
				seq = Xi_np[locus_idx]
				start, end = get_receptive_field_bounds(row, rows_per_locus, receptive_field, seq_length)
				node_sequences[rank].append(seq[:, start:end].copy())
		
		del Xi, Xi_np
	
	# Create PWMs for each ranked node
	pwm_results = {}
	for rank in range(num_top_nodes):
		node_idx = node_order[rank]
		sequences = node_sequences[rank]
		
		pwm = np.stack(sequences).mean(axis=0)
		activations = top_activations[rank]
		
		pwm_results[node_idx] = {
			'pwm': pwm,
			'rank': rank,
			'num_sequences': len(sequences),
			'top_magnitude': activations[0][0],
			'min_magnitude': activations[-1][0]
		}
	
	# === STEP 8: Save ===
	output_dir = out_dir
	os.makedirs(output_dir, exist_ok=True)
	
	pwm_file = os.path.join(output_dir, "pwms.npz")
	np.savez(pwm_file, **{f"node_{k}": v['pwm'] for k, v in pwm_results.items()})
	
	meta_out = {
		'layer': layer,
		'receptive_field': receptive_field,
		'num_top_nodes': num_top_nodes,
		'num_samples_per_node': num_samples_per_node,
		'seq_length': seq_length,
		'rows_per_locus': rows_per_locus,
		'pwm_file': pwm_file,
		'output_dir': output_dir,
		'node_info': {int(k): {
			'rank': int(v['rank']),
			'num_sequences': int(v['num_sequences']),
			'top_magnitude': float(v['top_magnitude']),
			'min_magnitude': float(v['min_magnitude'])
		} for k, v in pwm_results.items()}
	}
	
	with open(os.path.join(output_dir, "pwm_meta.json"), 'w') as f:
		json.dump(meta_out, f, indent=2)
	
	return pwm_results


def plot_node_pwms_from_meta(data_dir, max_nodes=None):
	"""Plot PWMs from metadata file."""
	
	# Raise error if pwm_meta.json does not exist in data_dir
	if not os.path.exists(os.path.join(data_dir, 'pwm_meta.json')):
		raise FileNotFoundError("pwm_meta.json does not exist in the current directory")
	
	with open(os.path.join(data_dir, 'pwm_meta.json')) as f:
		meta = json.load(f)
	
	output_dir = meta['output_dir']
	layer = meta['layer']
	
	logo_dir = os.path.join(output_dir, "logos")
	os.makedirs(logo_dir, exist_ok=True)
	
	# Load the PFM data and filter nodes
	data = np.load(meta['pwm_file'])
	sorted_nodes = list(meta['node_info'].items())
	if max_nodes:
		sorted_nodes = sorted_nodes[:max_nodes]
	
	# Individual plots
	for node_idx, info in tqdm(sorted_nodes, desc="Plotting logos"):
		rank = info['rank']
		pfm = data[f"node_{node_idx}"]
		
		# Add small pseudocount to avoid division by zero
		pseudocount = 0.0001
		pfm_with_pseudo = pfm + pseudocount
		
		# Normalize to get Position Probability Matrix (PPM)
		ppm = pfm_with_pseudo / pfm_with_pseudo.sum(axis=0, keepdims=True)
		
		# Calculate information content for each position
		# IC = log2(4) - entropy = 2 - entropy for DNA
		entropy = -np.sum(ppm * np.log2(ppm), axis=0)
		ic = 2.0 - entropy  # Maximum 2 bits for DNA
		
		# Scale PPM by information content for sequence logo
		logo_matrix = ppm * ic
		
		# Plot the logo
		fig, ax = plt.subplots(figsize=(12, 3))
		plot_logo(logo_matrix, ax=ax)
		ax.set_title(f'Node {node_idx} (rank {rank}) - Layer {layer}')
		ax.set_ylabel('Information content (bits)')
		plt.tight_layout()
		plt.savefig(os.path.join(logo_dir, f'node_{node_idx}_rank_{rank}.png'), dpi=150, bbox_inches='tight')
		plt.close()
	
	# Grid plot
	top_n = min(20, len(sorted_nodes))
	ncols = 4
	nrows = int(np.ceil(top_n / ncols))
	
	fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows*2.5))
	axes = axes.flatten() if nrows > 1 else [axes]
	
	for idx, (node_idx_str, info) in enumerate(sorted_nodes[:top_n]):
		pfm = data[f"node_{int(node_idx_str)}"]
		
		# Apply same transformation for grid plot
		pseudocount = 0.0001
		pfm_with_pseudo = pfm + pseudocount
		ppm = pfm_with_pseudo / pfm_with_pseudo.sum(axis=0, keepdims=True)
		entropy = -np.sum(ppm * np.log2(ppm), axis=0)
		ic = 2.0 - entropy
		logo_matrix = ppm * ic
		
		plot_logo(logo_matrix, ax=axes[idx])
		axes[idx].set_title(f'Node {node_idx_str} (rank {info["rank"]})', fontsize=10)
	
	for idx in range(top_n, len(axes)):
		axes[idx].axis('off')
	
	plt.suptitle(f'Top {top_n} Nodes - Layer {layer}', fontsize=16)
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f'top{top_n}_grid.png'), dpi=150, bbox_inches='tight')
	plt.close()