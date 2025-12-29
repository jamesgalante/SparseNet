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

		# Skip nodes with no sequences
		if len(sequences) == 0:
			print(f"Warning: No sequences found for node {node_idx} (rank {rank}). Skipping.")
			continue
		
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


# def plot_node_pwms_from_meta(data_dir, max_nodes=None):
# 	"""Plot PWMs from metadata file."""
# 	
# 	# Raise error if pwm_meta.json does not exist in data_dir
# 	if not os.path.exists(os.path.join(data_dir, 'pwm_meta.json')):
# 		raise FileNotFoundError("pwm_meta.json does not exist in the current directory")
# 	
# 	with open(os.path.join(data_dir, 'pwm_meta.json')) as f:
# 		meta = json.load(f)
# 	
# 	output_dir = meta['output_dir']
# 	layer = meta['layer']
# 	
# 	# Load the PFM data and filter nodes
# 	data = np.load(meta['pwm_file'])
# 	sorted_nodes = list(meta['node_info'].items())
# 	if max_nodes:
# 		sorted_nodes = sorted_nodes[:max_nodes]
# 	
# 	# Grid plot - show all nodes (or max_nodes if specified)
# 	total_nodes = len(sorted_nodes)
# 	ncols = 4
# 	nrows = int(np.ceil(total_nodes / ncols))
# 	
# 	fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows*2.5))
# 	axes = axes.flatten()
# 	
# 	for idx, (node_idx_str, info) in enumerate(tqdm(sorted_nodes, desc="Plotting grid")):
# 		pfm = data[f"node_{int(node_idx_str)}"]
# 		
# 		# Apply transformation for grid plot
# 		pseudocount = 0.0001
# 		pfm_with_pseudo = pfm + pseudocount
# 		ppm = pfm_with_pseudo / pfm_with_pseudo.sum(axis=0, keepdims=True)
# 		entropy = -np.sum(ppm * np.log2(ppm), axis=0)
# 		ic = 2.0 - entropy
# 		logo_matrix = ppm * ic
# 		
# 		plot_logo(logo_matrix, ax=axes[idx])
# 		axes[idx].set_title(f'Node {node_idx_str} (rank {info["rank"]})', fontsize=10)
# 	
# 	# Turn off any unused subplots
# 	for idx in range(total_nodes, len(axes)):
# 		axes[idx].axis('off')
# 	
# 	plt.suptitle(f'All {total_nodes} Nodes - Layer {layer}', fontsize=16)
# 	plt.tight_layout()
# 	plt.show()

def plot_node_pwms_from_meta(data_dir, max_nodes=None, save_path=None, show=True):
    """Plot PWMs from metadata file; optionally save to PNG instead of showing."""
    if not os.path.exists(os.path.join(data_dir, "pwm_meta.json")):
        raise FileNotFoundError("pwm_meta.json does not exist in the current directory")

    with open(os.path.join(data_dir, "pwm_meta.json")) as f:
        meta = json.load(f)

    layer = meta["layer"]

    # Load the PFM data and filter nodes
    data = np.load(meta["pwm_file"])
    sorted_nodes = list(meta["node_info"].items())
    if max_nodes:
        sorted_nodes = sorted_nodes[:max_nodes]

    # Grid plot - show all nodes (or max_nodes if specified)
    total_nodes = len(sorted_nodes)
    ncols = 4
    nrows = int(np.ceil(total_nodes / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 2.5))
    axes = axes.flatten()

    for idx, (node_idx_str, info) in enumerate(
        tqdm(sorted_nodes, desc="Plotting grid")
    ):
        pfm = data[f"node_{int(node_idx_str)}"]

        # Apply transformation for grid plot
        pseudocount = 0.0001
        pfm_with_pseudo = pfm + pseudocount
        ppm = pfm_with_pseudo / pfm_with_pseudo.sum(axis=0, keepdims=True)
        entropy = -np.sum(ppm * np.log2(ppm), axis=0)
        ic = 2.0 - entropy
        logo_matrix = ppm * ic

        plot_logo(logo_matrix, ax=axes[idx])
        axes[idx].set_title(f"Node {node_idx_str} (rank {info['rank']})", fontsize=10)

    # Turn off any unused subplots
    for idx in range(total_nodes, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"All {total_nodes} Nodes - Layer {layer}", fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

def compute_pwms_for_all_layers(
    loader,
    activations_dir,
    pwm_root_dir,
    latent_dim,
    num_top_nodes,
    num_samples_per_node,
):
    """
    Convenience function: compute PWMs for all SAE layers and save a grid figure
    for each layer in its own directory.

    Layout on disk:
        pwm_root_dir/
            layer0/
                pwms.npz
                pwm_meta.json
                pwm_grid.png
            layer1/
                ...
    """
    os.makedirs(pwm_root_dir, exist_ok=True)

    meta_path = os.path.join(activations_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    
    layer_names = meta["layers"]
    num_layers = len(layer_names)
    
    print(f"Computing PWMs for {num_layers} layers into {pwm_root_dir}")

    for layer_idx in range(num_layers):
        # Your receptive-field convention
        receptive_field = 21 if layer_idx == 0 else 21 + 2 ** (layer_idx + 1)

        layer_dir = os.path.join(pwm_root_dir, f"layer{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)

        print(f"\n=== Layer {layer_idx} ({layer_names[layer_idx]}) ===")
        pwm_results = create_PWM_for_top_N_nodes(
            layer=layer_idx,
            receptive_field=receptive_field,
            num_top_nodes=num_top_nodes,
            data_dir=activations_dir,
            out_dir=layer_dir,
            loader=loader,
            latent_dim=latent_dim,
            num_samples_per_node=num_samples_per_node,
        )

        print(
            f"  Saved {len(pwm_results)} PWMs for layer {layer_idx} to {layer_dir}"
        )

        # Save a grid image of all PWMs for this layer
        grid_path = os.path.join(layer_dir, "pwm_grid.png")
        plot_node_pwms_from_meta(
            data_dir=layer_dir,
            max_nodes=None,
            save_path=grid_path,
            show=False,
        )
        print(f"  Saved PWM grid to {grid_path}")
