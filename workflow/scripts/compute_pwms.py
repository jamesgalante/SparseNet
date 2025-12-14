import os

from pipeline_scripts import deterministic_data_loaders as ddl
from pipeline_scripts import create_PWM_for_nodes as pwm

inp = snakemake.input
out = snakemake.output
params = snakemake.params

os.makedirs(params.pwm_root_dir, exist_ok=True)

p = params.pwm_params
num_top_nodes = p["num_top_nodes"]
num_samples_per_node = p["num_samples_per_node"]
expansion_factor = p["expansion_factor"]

latent_dim = int(params.latent_base_channels * expansion_factor)

sae_testing_data = ddl.DeterministicPeakGenerator(
  peaks=[inp.peaks, inp.negatives],
  sequences=inp.seqs,
  signals=[inp.signal_plus, inp.signal_minus],
  chroms=None,
  in_window=2114,
  out_window=1000,
  pin_memory=True,
  batch_size=64,
  verbose=True,
)

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
