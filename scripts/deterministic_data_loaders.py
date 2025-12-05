# These classes are meant to create a simple data loader that cycles through all training peaks in their original order in the bed file
# When passing data through the SAEs, keeping the original order allows us to easily pair activations with motif presence from finemo outputs, which are similarly ordered

from tangermeme.io import extract_loci
import torch

# Simple Data Generator for Deterministic Peak Loading
def DeterministicPeakGenerator(peaks, sequences, signals,
	chroms=None, in_window=2114, out_window=1000, pin_memory=True, 
	batch_size=64, verbose=True):

	X_peaks = extract_loci(loci=peaks, sequences=sequences, 
		signals=signals, in_signals=None, chroms=chroms, in_window=in_window, 
		out_window=out_window, max_jitter=0, min_counts=None,
		max_counts=None, summits=False, exclusion_lists=None,
		ignore=list('QWERYUIOPSDFHJKLZXVBNM'), return_mask=True, verbose=verbose)

	X_gen = DeterministicPeakSampler(
		peak_sequences=X_peaks[0],
		peak_signals=X_peaks[1],
		in_window=in_window,
		out_window=out_window
	)

	X_gen = torch.utils.data.DataLoader(X_gen, pin_memory=pin_memory,
		num_workers=0, batch_size=batch_size, shuffle=False) 

	return X_gen

# Sample torch Dataset for loading peaks
class DeterministicPeakSampler(torch.utils.data.Dataset):
	def __init__(self, peak_sequences, peak_signals, in_window=2114, out_window=1000):

		self.peak_sequences = peak_sequences.numpy(force=True)
		self.peak_signals = peak_signals.numpy(force=True)
		self.in_window = in_window
		self.out_window = out_window
				
	def __len__(self):
		return len(self.peak_sequences)
		
	def __getitem__(self, idx):
				
		Xi = torch.from_numpy(self.peak_sequences[idx][:, :self.in_window])
		yi = torch.from_numpy(self.peak_signals[idx][:, :self.out_window])
			
		return Xi, yi, 1  # (sequence, signal, label)
