import os, sys

THIS_DIR = os.path.dirname(__file__)   # workflow/scripts
sys.path.insert(0, THIS_DIR)

from pipeline_scripts import deterministic_data_loaders as ddl
from pipeline_scripts import create_PWM_for_nodes as pwm

inp = snakemake.input
out = snakemake.output
params = snakemake.params
wc = snakemake.wildcards

os.makedirs(params.pwm_root_dir, exist_ok=True)

expansion_factor = float(wc.ef)
num_samples_per_node = int(wc.ns)

latent_dim = int(params.latent_base_channels * expansion_factor)
num_top_nodes = latent_dim  # per your “64 * expansion_factor” convention

sae_testing_data = ddl.DeterministicPeakGenerator(
  peaks=[inp.peaks, inp.negatives],
  sequences=inp.seqs,
  signals=[inp.signal_plus, inp.signal_minus],
  chroms=None,
  in_window=2114,
  out_window=1000,
  pin_memory=False,
  batch_size=64,
  verbose=True,
)

# NOTE: you previously removed trainer dependence; this call assumes your
# compute_pwms_for_all_layers signature does NOT require trainer.
pwm.compute_pwms_for_all_layers(
  loader=sae_testing_data,
  activations_dir=params.activations_dir,
  pwm_root_dir=params.pwm_root_dir,
  latent_dim=latent_dim,
  num_top_nodes=num_top_nodes,
  num_samples_per_node=num_samples_per_node,
)

with open(out.done, "w") as f:
  f.write("done\n")
