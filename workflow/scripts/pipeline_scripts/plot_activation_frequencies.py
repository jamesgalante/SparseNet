# Plot node activation frequency
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_node_activation_frequencies(num_layers: int, latent_dim: int, data_dir: str) -> None:

    # Get batch file paths 
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    batch_files = meta["batch_files"]

    # Count node appearances per layer
    counts_per_layer = np.zeros((num_layers, latent_dim), dtype=np.int64)
    for path in batch_files:
        arr = np.load(path, mmap_mode="r")
        idx = arr[:, :, :, 0].astype(np.int32, copy=False)
        for l in range(num_layers):
            counts_per_layer[l] += np.bincount(idx[:, l, :].ravel(), minlength=latent_dim)

    plt.figure(figsize=(10, 6))
    for l in range(num_layers):
        sorted_counts = np.sort(counts_per_layer[l])[::-1]
        plt.plot(sorted_counts, label=f"Layer {l}")

    plt.xlabel("Node rank (sorted by frequency)")
    plt.ylabel("Count")
    plt.yscale('log')
    plt.title("Node Activation Frequency per Layer")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(data_dir, "node_activation_frequency_per_layer.png")
    plt.savefig(save_path, dpi=300)

    plt.close()
