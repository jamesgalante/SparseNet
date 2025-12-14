import os, sys

THIS_DIR = os.path.dirname(__file__)   # workflow/scripts
sys.path.insert(0, THIS_DIR)

from bpnetlite.io import PeakGenerator
from pipeline_scripts import deterministic_data_loaders as ddl
from pipeline_scripts import models as mds
from pipeline_scripts import SAE_trainer as st
from pipeline_scripts import save_activations as sa
from pipeline_scripts import plot_activation_frequencies as nf

inp = snakemake.input
out = snakemake.output
params = snakemake.params
wc = snakemake.wildcards

os.makedirs(params.out_dir, exist_ok=True)

# hyperparams from wildcards (strings) -> cast
center_len = int(wc.cl)
expansion_factor = float(wc.ef)
topk_fraction = float(wc.tf)

# training constants from params/config
epochs = int(params.epochs)
inner_bs = int(params.inner_bs)
lr = float(params.lr)

training_data = PeakGenerator(
  peaks=inp.peaks,
  negatives=inp.negatives,
  sequences=inp.seqs,
  signals=[inp.signal_plus, inp.signal_minus],
  controls=[inp.ctl_plus, inp.ctl_minus],
  chroms=None,
  in_window=2114,
  out_window=1000,
  max_jitter=128,
  negative_ratio=0.33,
  reverse_complement=True,
  shuffle=True,
  min_counts=None,
  max_counts=None,
  summits=False,
  exclusion_lists=None,
  random_state=12345,
  pin_memory=True,
  num_workers=0,
  batch_size=64,
  verbose=True,
)

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

trainer = st.SAETrainer(
  model_path=inp.model_path,
  device="cuda",
  center_len=center_len,
)

trainer.train_all(
  train_loader=training_data,
  sae_cls=mds.SAETopK,
  sae_kwargs={"latent_multiplier": expansion_factor, "k_fraction": topk_fraction},
  save_dir=os.path.join(params.out_dir, "models"),
  logs_dir=os.path.join(params.out_dir, "models", "logs"),
  epochs=epochs,
  inner_bs=inner_bs,
  lr=lr,
  log_every=50,
)

sae_models = mds.load_saes_from_dir(
  save_dir=os.path.join(params.out_dir, "models"),
  layers=trainer.layers,
  device=trainer.device,
)

# should write activations/meta.json (your downstream target)
sa.collect_topk_indices_to_disk_from_trainer(
  trainer=trainer,
  sae_models=sae_models,
  loader=sae_testing_data,
  out_dir=os.path.join(params.out_dir, "activations"),
)

# should write activations/node_activation_frequencies.png (your declared output)
nf.plot_node_activation_frequencies(
  num_layers=len(trainer.layers),
  latent_dim=int(expansion_factor * 64),
  data_dir=os.path.join(params.out_dir, "activations"),
)
