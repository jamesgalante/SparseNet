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
    def plot_sample(self, sample_idx, layer_idx=None, top_n_nodes=50, dense_yticks=False):
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
        
        # --- Toggle for dense vs sparse y-axis ticks ---
        if dense_yticks:
            # Show all node indices
            ax2.set_yticks(np.arange(top_n_nodes))
            ax2.set_yticklabels(
                [f"{top_node_indices[i]}" for i in range(top_n_nodes)],
                fontsize=8
            )
        else:
            # Show ~10 evenly spaced ticks (original behavior)
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

        # print(
        #     f"\nShowing all nodes with |activation| > {threshold} "
        #     f"for positions {start}-{end}:"
        # )
        # for pos, nodes_for_pos in zip(positions, pos_nodes):
        #     if not nodes_for_pos:
        #         print(f"  Position {pos}: no activated nodes.")
        #         continue
        #     print(f"  Position {pos}:")
        #     for node_idx, act_val in nodes_for_pos:
        #         node_info = self.pwm_meta["node_info"].get(str(node_idx), {})
        #         rank = node_info.get("rank", "N/A")
        #         print(
        #             f"    Node {node_idx:4d} (rank {rank}): "
        #             f"activation = {act_val:.4f}"
        #         )

    def show_position_range_all_layers(self, sample_idx, start_pos, end_pos, 
                                    pwm_root_dir, threshold=0.0, max_nodes_per_pos=3):
        """
        Show activated nodes across ALL layers for a range of positions.
        
        Parameters
        ----------
        sample_idx : int
            Sample to visualize
        start_pos : int
            Start position (0-based, relative to activation window)
        end_pos : int
            End position (inclusive)
        pwm_root_dir : str
            Root directory containing layer0/, layer1/, etc. subdirectories with PWMs
        threshold : float
            Minimum |activation| to include
        max_nodes_per_pos : int
            Maximum number of nodes to show per position (top by activation)
        """
        # Load sample data if not already loaded
        if self.current_sample_idx != sample_idx:
            print(f"Loading sample {sample_idx}...")
            Xi, yi, activations = self.load_sample_data(sample_idx)
            self.current_Xi = Xi
            self.current_signal = yi
            self.current_activations = activations
            self.current_sample_idx = sample_idx
        
        # Load PWM data for ALL layers
        pwm_data_per_layer = {}
        pwm_meta_per_layer = {}
        
        for layer_idx in range(self.num_layers):
            layer_dir = os.path.join(pwm_root_dir, f'layer{layer_idx}')
            if not os.path.exists(layer_dir):
                continue
            
            meta_path = os.path.join(layer_dir, 'pwm_meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                    pwm_path = meta.get('pwm_file')
                    if pwm_path and os.path.exists(pwm_path):
                        pwm_data_per_layer[layer_idx] = np.load(pwm_path)
                        pwm_meta_per_layer[layer_idx] = meta
        
        print(f"Loaded PWMs for {len(pwm_data_per_layer)} layers")
        
        # Validate position range
        start = max(0, start_pos)
        end = min(self.rows_per_locus - 1, end_pos)
        if start > end:
            print(f"Error: invalid range after clipping: {start}-{end}.")
            return
        
        positions = list(range(start, end + 1))
        n_positions = len(positions)
        
        # Build dense activations and collect nodes for each layer
        layer_pos_nodes = {}
        
        for layer_idx in sorted(pwm_data_per_layer.keys()):
            layer_acts = self.current_activations[:, layer_idx, :, :]
            
            # Build dense activation matrix
            max_idx_acts = int(layer_acts[:, :, 0].max())
            pwm_node_indices = [
                int(k.split('_')[1]) 
                for k in pwm_data_per_layer[layer_idx].keys() 
                if k.startswith('node_')
            ]
            max_idx_pwms = max(pwm_node_indices) if pwm_node_indices else -1
            latent_dim = max(max_idx_acts, max_idx_pwms) + 1
            
            dense_acts = np.zeros((self.rows_per_locus, latent_dim), dtype=np.float32)
            for pos in range(self.rows_per_locus):
                indices = layer_acts[pos, :, 0].astype(int)
                values = layer_acts[pos, :, 1]
                dense_acts[pos, indices] = values
            
            # Collect nodes for each position
            layer_pos_nodes[layer_idx] = []
            for pos in positions:
                act = dense_acts[pos]
                mask = np.abs(act) > threshold
                node_idx = np.where(mask)[0]
                
                if node_idx.size == 0:
                    layer_pos_nodes[layer_idx].append([])
                    continue
                
                vals = np.abs(act[node_idx])
                order = np.argsort(vals)[::-1][:max_nodes_per_pos]
                nodes_for_pos = list(zip(node_idx[order], vals[order]))
                layer_pos_nodes[layer_idx].append(nodes_for_pos)
        
        # Create figure: rows = layers, cols = positions
        n_layers_with_pwms = len(layer_pos_nodes)
        if n_layers_with_pwms == 0:
            print("No layers with PWMs available!")
            return
        
        fig, axes = plt.subplots(
            n_layers_with_pwms, n_positions,
            figsize=(3.5 * n_positions, 3 * n_layers_with_pwms)
        )
        
        if n_layers_with_pwms == 1 and n_positions == 1:
            axes = np.array([[axes]])
        elif n_layers_with_pwms == 1:
            axes = axes.reshape(1, n_positions)
        elif n_positions == 1:
            axes = axes.reshape(n_layers_with_pwms, 1)
        
        # Plot each layer × position
        layer_row = 0
        for layer_idx in sorted(layer_pos_nodes.keys()):
            pos_nodes = layer_pos_nodes[layer_idx]
            pwm_data = pwm_data_per_layer[layer_idx]
            
            for col, pos in enumerate(positions):
                ax = axes[layer_row, col]
                nodes_for_pos = pos_nodes[col]
                
                if len(nodes_for_pos) == 0:
                    ax.text(0.5, 0.5, 'No activation', 
                        ha='center', va='center', fontsize=9)
                    ax.axis('off')
                    if col == 0:
                        ax.set_ylabel(f'L{layer_idx}', fontsize=10, fontweight='bold')
                    if layer_row == 0:
                        ax.set_title(f'Pos {pos}', fontsize=10, fontweight='bold')
                    continue
                
                # Show most activated node's logo
                node_idx, act_val = nodes_for_pos[0]
                pwm_key = f'node_{node_idx}'
                
                if pwm_key not in pwm_data:
                    ax.text(0.5, 0.5, f'N{node_idx}\nNo PWM', 
                        ha='center', va='center', fontsize=8)
                    ax.axis('off')
                else:
                    pfm = pwm_data[pwm_key]
                    pseudocount = 1e-4
                    pfm_pseudo = pfm + pseudocount
                    ppm = pfm_pseudo / pfm_pseudo.sum(axis=0, keepdims=True)
                    entropy = -np.sum(ppm * np.log2(ppm), axis=0)
                    ic = 2.0 - entropy
                    logo_matrix = ppm * ic
                    plot_logo(logo_matrix, ax=ax)
                
                node_info_str = f'N{node_idx} ({act_val:.2f})'
                if len(nodes_for_pos) > 1:
                    node_info_str += f'\n+{len(nodes_for_pos)-1}'
                
                ax.set_title(node_info_str, fontsize=8)
                
                if col == 0:
                    ax.set_ylabel(f'L{layer_idx}', fontsize=10, fontweight='bold')
                if layer_row == 0:
                    ax.set_title(f'Pos {pos}\n{node_info_str}', fontsize=8)
            
            layer_row += 1
        
        plt.suptitle(
            f'All layers: sample {sample_idx}, positions {start}-{end}',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()

    def _get_node_idx_from_rank(self, rank):
        """Convert rank to node index using pwm_meta"""
        for node_idx_str, info in self.pwm_meta["node_info"].items():
            if info.get("rank") == rank:
                return int(node_idx_str)
        raise ValueError(f"No node found with rank {rank}")

    def plot_node_activation_traces(self, node_rank=None, node_idx=None, 
                                     layer_idx=None, n_samples=100, 
                                     alpha=0.3, colormap='viridis', verbose=True):
        """
        Plot activation magnitude across positions for a specific node 
        across multiple samples, with PWM logo on the left.
        
        Parameters
        ----------
        node_rank : int, optional
            Rank of the node (from pwm_meta.json). Either this or node_idx required.
        node_idx : int, optional
            Direct node index. Either this or node_rank required.
        layer_idx : int or None
            Which SAE layer to visualize. If None, uses the PWM layer.
        n_samples : int
            Number of samples to plot (default 100)
        alpha : float
            Transparency for individual traces (default 0.3)
        colormap : str
            Matplotlib colormap name for coloring traces (default 'viridis')
        verbose : bool
            If True, print progress and statistics (default True)
        """
        # Determine layer
        if layer_idx is None:
            layer_idx = self.pwm_layer_idx
        self._check_layer_alignment(layer_idx)

        # Determine node index
        if node_idx is None and node_rank is None:
            raise ValueError("Must provide either node_rank or node_idx")
        if node_idx is None:
            node_idx = self._get_node_idx_from_rank(node_rank)
        
        # Get node info
        node_info = self.pwm_meta["node_info"].get(str(node_idx), {})
        rank = node_info.get("rank", "N/A")
        
        # Get PWM
        pwm_key = f"node_{node_idx}"
        if pwm_key not in self.pwm_data:
            raise ValueError(f"Node {node_idx} has no PWM available")
        
        pfm = self.pwm_data[pwm_key]
        
        # Convert PFM to information content logo
        pseudocount = 1e-4
        pfm_pseudo = pfm + pseudocount
        ppm = pfm_pseudo / pfm_pseudo.sum(axis=0, keepdims=True)
        entropy = -np.sum(ppm * np.log2(ppm), axis=0)
        ic = 2.0 - entropy
        logo_matrix = ppm * ic

        # Collect activation traces across samples
        if verbose:
            print(f"Collecting activation traces for node {node_idx} (rank {rank})...")
        traces = []
        valid_samples = 0
        
        # Count total samples available
        total_samples = len(self.loader.dataset)
        n_samples = min(n_samples, total_samples)
        
        for sample_idx in range(n_samples):
            try:
                _, _, activations = self.load_sample_data(sample_idx)
                layer_acts = activations[:, layer_idx, :, :]
                
                # Build dense activation vector for this sample
                max_idx_acts = int(layer_acts[:, :, 0].max())
                latent_dim = max(max_idx_acts, node_idx) + 1
                
                dense_acts = np.zeros(self.rows_per_locus, dtype=np.float32)
                for pos in range(self.rows_per_locus):
                    indices = layer_acts[pos, :, 0].astype(int)
                    values = layer_acts[pos, :, 1]
                    mask = indices == node_idx
                    if mask.any():
                        dense_acts[pos] = values[mask][0]
                
                traces.append(dense_acts)
                valid_samples += 1
                
            except Exception as e:
                print(f"Warning: Could not load sample {sample_idx}: {e}")
                continue
        
        if not traces:
            print("No valid samples found!")
            return
        
        traces = np.array(traces)  # shape: (n_samples, rows_per_locus)
        if verbose:
            print(f"Loaded {valid_samples} samples")
        
        # Create figure with PWM on left, traces on right
        fig = plt.figure(figsize=(18, 2.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[0.6, 4], wspace=0.15)
        
        ax_pwm = fig.add_subplot(gs[0])
        ax_traces = fig.add_subplot(gs[1])
        
        # Plot PWM logo
        plot_logo(logo_matrix, ax=ax_pwm)
        ax_pwm.set_title(
            f"Node {node_idx}\n(rank {rank})",
            fontsize=10,
            fontweight='bold'
        )
        
        # Plot activation traces
        positions = np.arange(self.rows_per_locus)
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i / valid_samples) for i in range(valid_samples)]
        
        # Plot individual traces
        for i, trace in enumerate(traces):
            ax_traces.plot(positions, trace, alpha=alpha, 
                          color=colors[i], linewidth=0.8)
        
        # Plot mean trace
        mean_trace = traces.mean(axis=0)
        ax_traces.plot(positions, mean_trace, color='black', 
                      linewidth=2.5, label='Mean', zorder=100)
        
        ax_traces.set_xlabel('Position (relative to activation window)', fontsize=11)
        ax_traces.set_ylabel('Activation magnitude', fontsize=11)
        ax_traces.set_title(
            f'Activation traces across {valid_samples} samples (Layer {layer_idx}: {self.layer_names[layer_idx]})',
            fontsize=11,
            fontweight='bold'
        )
        ax_traces.grid(True, alpha=0.3)
        ax_traces.legend(loc='upper right')
        ax_traces.set_xlim(0, self.rows_per_locus - 1)
        
        # Add stats box
        pct_active = (traces.max(axis=1) > 0).mean() * 100
        stats_text = f"% samples active: {pct_active:.1f}%"
        ax_traces.text(
            0.02, 0.98, stats_text,
            transform=ax_traces.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9
        )
        
        plt.tight_layout()
        plt.show()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Node {node_idx} (rank {rank}) - Summary Statistics")
            print(f"{'='*70}")
            print(f"Samples analyzed: {valid_samples}")
            print(f"Active in {(traces.max(axis=1) > 0).sum()}/{valid_samples} samples")
            print(f"Mean activation (all positions): {mean_trace.mean():.4f}")
            print(f"Max activation (any sample): {traces.max():.4f}")