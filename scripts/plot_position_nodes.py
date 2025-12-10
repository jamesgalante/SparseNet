import numpy as np
import matplotlib.pyplot as plt
from tangermeme.plot import plot_logo
import json
import os


class SequenceNodeVisualizer:
    """
    Visualize genomic sequences with corresponding SAE node activations
    using pre-computed activation data and layer-specific PWMs.

    Assumptions:
      - activations_dir contains:
          * meta.json with keys: layers, rows_per_locus, batch_files, ...
          * rows_batch_XXXXX.npy with shape
              (batch_size * rows_per_locus, num_layers, k, 2)
            where [:, :, :, 0] are node indices, [:, :, :, 1] are values.
      - pwm_dir contains (for ONE layer):
          * pwm_meta.json with key 'layer' (int layer index)
          * pwms.npz with arrays named 'node_{idx}' for each node.
    """

    def __init__(self, activations_dir, pwm_dir, loader):
        """
        Parameters
        ----------
        activations_dir : str
            Directory containing saved activation .npy files and meta.json
        pwm_dir : str
            Directory containing pwm_meta.json and pwms.npz files for ONE layer
        loader : DataLoader
            DataLoader for getting sequences and signals in the SAME order
            used to generate the activations.
        """
        self.loader = loader

        # --- Load activation metadata ---
        with open(os.path.join(activations_dir, "meta.json")) as f:
            self.act_meta = json.load(f)
        self.activations_dir = activations_dir
        self.batch_files = self.act_meta["batch_files"]
        self.rows_per_locus = self.act_meta["rows_per_locus"]
        self.layer_names = self.act_meta["layers"]
        self.num_layers = len(self.layer_names)

        # --- Load PWM metadata for a SINGLE layer ---
        with open(os.path.join(pwm_dir, "pwm_meta.json")) as f:
            self.pwm_meta = json.load(f)
        self.pwm_dir = pwm_dir
        self.pwm_data = np.load(self.pwm_meta["pwm_file"])
        # Layer index (integer) for which these PWMs were computed
        self.pwm_layer_idx = int(self.pwm_meta["layer"])

        # Current state (set by plot_sample)
        self.current_sample_idx = None
        self.current_Xi = None
        self.current_signal = None
        self.current_activations = None
        self.current_layer_idx = None
        self.current_dense_acts = None
        self.current_top_nodes = None
        self.current_center_offset = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _check_layer_alignment(self, layer_idx):
        """
        Ensure that the layer we are visualizing matches the layer
        for which PWMs were computed.
        """
        if layer_idx != self.pwm_layer_idx:
            raise ValueError(
                f"PWMs in '{self.pwm_dir}' were computed for layer_idx="
                f"{self.pwm_layer_idx}, but you requested layer_idx={layer_idx}. "
                "Either (1) regenerate PWMs for this layer using "
                "`create_PWM_for_top_N_nodes(layer=layer_idx, ...)` and point "
                "pwm_dir to that directory, or (2) call plot_sample with "
                "layer_idx=None to use the PWM layer."
            )

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    def load_sample_data(self, sample_idx):
        """
        Load sequence, signal, and activations for a specific sample.

        Returns
        -------
        Xi_sample : np.ndarray
            One-hot sequence, shape (4, in_window)
        yi_sample : np.ndarray
            Signal, shape (2, out_window)
        sample_acts : np.ndarray
            Activations for this sample, shape
            (rows_per_locus, num_layers, k, 2)
        """
        batch_size = self.loader.batch_size
        batch_idx = sample_idx // batch_size
        offset = sample_idx % batch_size

        # 1) Sequence + signal from loader (same order as activations)
        Xi_sample = None
        yi_sample = None
        for idx, batch in enumerate(self.loader):
            if idx == batch_idx:
                Xi, yi, *_ = batch
                Xi_sample = Xi[offset].numpy()
                yi_sample = yi[offset].numpy()
                break
        if Xi_sample is None:
            raise IndexError(f"Sample index {sample_idx} out of range.")

        # 2) Activations from .npy shard
        act_file = self.batch_files[batch_idx]
        act_data = np.load(act_file, mmap_mode="r")
        # act_data shape: (batch_size * rows_per_locus, num_layers, k, 2)
        start_row = offset * self.rows_per_locus
        end_row = start_row + self.rows_per_locus
        sample_acts = act_data[start_row:end_row, :, :, :]

        return Xi_sample, yi_sample, sample_acts

    # ------------------------------------------------------------------ #
    # Main visualization: sample + heatmap
    # ------------------------------------------------------------------ #
    def plot_sample(self, sample_idx, layer_idx=None, top_n_nodes=50):
        """
        Plot a genomic sequence with its node activations as a heatmap.

        Parameters
        ----------
        sample_idx : int
            Index of sample to visualize
        layer_idx : int or None
            Which SAE layer to visualize (0-indexed in activations meta).
            If None, defaults to the PWM layer (self.pwm_layer_idx).
        top_n_nodes : int
            Number of most active nodes to show in the heatmap.
        """
        # Decide which layer to use
        if layer_idx is None:
            layer_idx = self.pwm_layer_idx
        self._check_layer_alignment(layer_idx)

        # Load sample data
        self.current_sample_idx = sample_idx
        Xi, yi, activations = self.load_sample_data(sample_idx)
        self.current_Xi = Xi
        self.current_signal = yi
        self.current_activations = activations
        self.current_layer_idx = layer_idx

        # Extract activations for this layer: (rows_per_locus, k, 2)
        layer_acts = activations[:, layer_idx, :, :]

        # --- Build dense activation matrix (positions × latent_dim) ---
        # Max node index referenced in activations
        max_idx_acts = int(layer_acts[:, :, 0].max())

        # Max node index that has a PWM (usually same or smaller)
        pwm_node_indices = [
            int(k.split("_")[1])
            for k in self.pwm_data.keys()
            if k.startswith("node_")
        ]
        max_idx_pwms = max(pwm_node_indices) if pwm_node_indices else -1

        latent_dim = max(max_idx_acts, max_idx_pwms) + 1

        dense_acts = np.zeros((self.rows_per_locus, latent_dim), dtype=np.float32)
        for pos in range(self.rows_per_locus):
            indices = layer_acts[pos, :, 0].astype(int)  # node indices
            values = layer_acts[pos, :, 1]               # activation values
            dense_acts[pos, indices] = values

        # Node ranking by max |activation| across positions
        node_max_act = np.abs(dense_acts).max(axis=0)
        top_node_indices = np.argsort(node_max_act)[-top_n_nodes:][::-1]

        # ------------------------------------------------------------------
        # Top panel: signal over full 2114-bp input
        # ------------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 10),
            gridspec_kw={"height_ratios": [1, 3]}
        )

        full_seq_len = Xi.shape[-1]
        signal_len = yi.shape[-1]

        signal_offset = (full_seq_len - signal_len) // 2
        positions_signal = np.arange(signal_len) + signal_offset

        # Plot plus and minus signal in the central 1000-bp window
        ax1.fill_between(
            positions_signal, 0, yi[0],
            alpha=0.7, label="Plus strand", color="blue"
        )
        ax1.fill_between(
            positions_signal, 0, -yi[1],
            alpha=0.7, label="Minus strand", color="red"
        )

        # Activation window highlight (rows_per_locus positions)
        center_offset = (full_seq_len - self.rows_per_locus) // 2
        ax1.axvspan(
            center_offset,
            center_offset + self.rows_per_locus,
            alpha=0.1, color="gray",
            label=f"Activation window ({self.rows_per_locus} bp)",
        )

        ax1.set_ylabel("Signal", fontsize=12)
        ax1.set_xlim(0, full_seq_len)
        ax1.legend(loc="upper right")
        ax1.set_title(
            f"Sample {sample_idx} - Genomic Signal and Node Activations "
            f"(Layer {layer_idx}: {self.layer_names[layer_idx]})",
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)

        # ------------------------------------------------------------------
        # Bottom panel: node-activation heatmap in the activation window
        # ------------------------------------------------------------------
        act_subset = dense_acts[:, top_node_indices].T  # (top_n_nodes, positions)
        im = ax2.imshow(
            act_subset,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            origin="lower",
            extent=[center_offset,
                    center_offset + self.rows_per_locus,
                    0,
                    top_n_nodes],
        )

        # X-axis ticks in "activation-window coordinates" (0 .. rows_per_locus-1)
        num_ticks = 11
        tick_positions = np.linspace(
            center_offset,
            center_offset + self.rows_per_locus,
            num_ticks,
        )
        tick_labels = np.linspace(
            0,
            self.rows_per_locus,
            num_ticks,
            dtype=int,
        )

        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, fontsize=10)
        ax2.set_xlabel(
            f"Activation-window position (0–{self.rows_per_locus - 1})",
            fontsize=12,
        )

        ax2.set_ylabel("Node index (ranked by max activation)", fontsize=12)
        ax2.set_xlim(0, full_seq_len)  # align with top plot
        ax2.set_yticks(
            np.arange(0, top_n_nodes, max(1, top_n_nodes // 10))
        )
        ax2.set_yticklabels(
            [f"{top_node_indices[i]}"
             for i in range(0, top_n_nodes, max(1, top_n_nodes // 10))]
        )

        plt.tight_layout()
        plt.show()

        # Store for later logo plotting
        self.current_top_nodes = top_node_indices
        self.current_dense_acts = dense_acts
        self.current_center_offset = center_offset

        print("\nVisualization ready.")
        print(
            f"Activation window: positions {center_offset}-"
            f"{center_offset + self.rows_per_locus - 1} "
            f"(of {full_seq_len} total bp)"
        )
        print(
            f"Use positions 0-{self.rows_per_locus - 1} in "
            "show_position_logos/show_position_range_logos "
            "(relative to activation window)."
        )

    # ------------------------------------------------------------------ #
    # Single-position logo view
    # ------------------------------------------------------------------ #
    def show_position_logos(self, position, top_k=10, threshold=0.01):
        """
        Show PWM logos for nodes activated at a specific position.

        Parameters
        ----------
        position : int
            Position within activation window [0, rows_per_locus).
        top_k : int
            Maximum number of logos to show (sorted by |activation|).
        threshold : float
            Minimum |activation| to include.
        """
        if self.current_dense_acts is None:
            print("Error: No sample plotted yet. Call plot_sample() first.")
            return

        if position < 0 or position >= self.rows_per_locus:
            print(f"Error: position must be between 0 and {self.rows_per_locus - 1}.")
            return

        layer_idx = self.current_layer_idx
        self._check_layer_alignment(layer_idx)

        act = self.current_dense_acts[position]  # (latent_dim,)

        # Which nodes are "active" at this position?
        active_mask = np.abs(act) > threshold
        active_indices = np.where(active_mask)[0]
        if active_indices.size == 0:
            print(
                f"No nodes activated above threshold {threshold} "
                f"at position {position}."
            )
            return

        active_values = np.abs(act[active_indices])
        order = np.argsort(active_values)[::-1][:top_k]
        active_indices = active_indices[order]
        active_values = active_values[order]

        print("\n" + "=" * 70)
        print(
            f"Position {position} - Top {len(active_indices)} activated nodes "
            f"(threshold={threshold})"
        )
        print("=" * 70 + "\n")

        n_logos = len(active_indices)
        ncols = min(4, n_logos)
        nrows = int(np.ceil(n_logos / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        if n_logos == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (node_idx, act_val) in enumerate(zip(active_indices, active_values)):
            pwm_key = f"node_{node_idx}"
            ax = axes[idx]

            if pwm_key not in self.pwm_data:
                ax.text(
                    0.5,
                    0.5,
                    f"Node {node_idx}\nNo PWM available",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                ax.axis("off")
                continue

            pfm = self.pwm_data[pwm_key]

            # PFM -> information-content logo
            pseudocount = 1e-4
            pfm_pseudo = pfm + pseudocount
            ppm = pfm_pseudo / pfm_pseudo.sum(axis=0, keepdims=True)
            entropy = -np.sum(ppm * np.log2(ppm), axis=0)
            ic = 2.0 - entropy
            logo_matrix = ppm * ic

            plot_logo(logo_matrix, ax=ax)

            node_info = self.pwm_meta["node_info"].get(str(node_idx), {})
            rank = node_info.get("rank", "N/A")

            ax.set_title(
                f"Node {node_idx} (rank {rank})\n"
                f"Activation: {act_val:.3f}",
                fontsize=10,
                fontweight="bold",
            )

        # Hide unused axes
        for idx in range(n_logos, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Activated node motifs at position {position} "
            f"(layer {layer_idx})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        # print(f"\nShowing {len(active_indices)} nodes with |activation| > {threshold}")
        # for node_idx, act_val in zip(active_indices, active_values):
        #     node_info = self.pwm_meta["node_info"].get(str(node_idx), {})
        #     rank = node_info.get("rank", "N/A")
        #     print(
        #         f"  Node {node_idx:4d} (rank {rank}): "
        #         f"activation = {act_val:.4f}"
        #     )

    # ------------------------------------------------------------------ #
    # Multi-position grid: columns = positions, rows = activated nodes
    # ------------------------------------------------------------------ #
    def show_position_range_logos(self, start_pos, end_pos, threshold=0.0):
        """
        Show PWM logos for all activated nodes across a range of positions.

        Layout:
          - Each column is a genomic position (relative to activation window).
          - Each row within that column is one activated node at that position,
            sorted by |activation| (largest on top).

        Parameters
        ----------
        start_pos : int
            Start position (inclusive), 0-based within activation window.
        end_pos : int
            End position (inclusive), 0-based within activation window.
        threshold : float, optional
            Minimum |activation| to include. Default 0.0 -> all nonzero entries.
        """
        if self.current_dense_acts is None:
            print("Error: No sample plotted yet. Call plot_sample() first.")
            return

        if start_pos < 0 or end_pos < 0:
            print("Error: start_pos and end_pos must be >= 0.")
            return
        if start_pos >= self.rows_per_locus:
            print(f"Error: start_pos must be < {self.rows_per_locus}.")
            return

        start = max(0, start_pos)
        end = min(self.rows_per_locus - 1, end_pos)
        if start > end:
            print(f"Error: invalid range after clipping: {start}-{end}.")
            return

        layer_idx = self.current_layer_idx
        self._check_layer_alignment(layer_idx)

        positions = list(range(start, end + 1))
        n_positions = len(positions)

        pos_nodes = []  # list of lists: [(node_idx, act_val), ...] per position
        max_nodes = 0

        for pos in positions:
            act = self.current_dense_acts[pos]  # (latent_dim,)
            mask = np.abs(act) > threshold
            node_idx = np.where(mask)[0]
            if node_idx.size == 0:
                pos_nodes.append([])
                continue

            vals = np.abs(act[node_idx])
            order = np.argsort(vals)[::-1]

            node_idx_sorted = node_idx[order]
            vals_sorted = vals[order]

            nodes_for_pos = list(zip(node_idx_sorted, vals_sorted))
            pos_nodes.append(nodes_for_pos)
            max_nodes = max(max_nodes, len(nodes_for_pos))

        if max_nodes == 0:
            print(
                f"No nodes activated above threshold {threshold} "
                f"in positions {start}-{end}."
            )
            return

        fig, axes = plt.subplots(
            max_nodes,
            n_positions,
            figsize=(4 * n_positions, 3 * max_nodes),
        )

        if max_nodes == 1 and n_positions == 1:
            axes = np.array([[axes]])
        elif max_nodes == 1:
            axes = axes.reshape(1, n_positions)
        elif n_positions == 1:
            axes = axes.reshape(max_nodes, 1)

        for col, pos in enumerate(positions):
            nodes_for_pos = pos_nodes[col]

            for row in range(max_nodes):
                ax = axes[row, col]

                if row >= len(nodes_for_pos):
                    ax.axis("off")
                    continue

                node_idx, act_val = nodes_for_pos[row]
                pwm_key = f"node_{node_idx}"

                if pwm_key not in self.pwm_data:
                    ax.text(
                        0.5,
                        0.5,
                        f"Node {node_idx}\nNo PWM",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
                    ax.axis("off")
                    continue

                pfm = self.pwm_data[pwm_key]

                pseudocount = 1e-4
                pfm_pseudo = pfm + pseudocount
                ppm = pfm_pseudo / pfm_pseudo.sum(axis=0, keepdims=True)
                entropy = -np.sum(ppm * np.log2(ppm), axis=0)
                ic = 2.0 - entropy
                logo_matrix = ppm * ic

                plot_logo(logo_matrix, ax=ax)

                node_info = self.pwm_meta["node_info"].get(str(node_idx), {})
                rank = node_info.get("rank", "N/A")

                if row == 0:
                    ax.set_title(
                        f"pos {pos}\n"
                        f"node {node_idx} (rank {rank}) act {act_val:.3f}",
                        fontsize=9,
                        fontweight="bold",
                    )
                else:
                    ax.set_title(
                        f"node {node_idx} (rank {rank}) act {act_val:.3f}",
                        fontsize=9,
                    )

        plt.suptitle(
            f"Activated node motifs for positions {start}-{end} "
            f"(sample {self.current_sample_idx}, layer {layer_idx})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        print(
            f"\nShowing all nodes with |activation| > {threshold} "
            f"for positions {start}-{end}:"
        )
        for pos, nodes_for_pos in zip(positions, pos_nodes):
            if not nodes_for_pos:
                print(f"  Position {pos}: no activated nodes.")
                continue
            print(f"  Position {pos}:")
            for node_idx, act_val in nodes_for_pos:
                node_info = self.pwm_meta["node_info"].get(str(node_idx), {})
                rank = node_info.get("rank", "N/A")
                print(
                    f"    Node {node_idx:4d} (rank {rank}): "
                    f"activation = {act_val:.4f}"
                )
