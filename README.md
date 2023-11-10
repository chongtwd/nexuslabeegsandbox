## Nexus lab Dementia EEG sandbox

### Data
Current data source is from [openneuro](https://openneuro.org/datasets/ds004504/versions/1.0.6). Containing around 88 patients with a mix of healthy and dementia patients.

###
Function of scripts so far

1. eda.py - generates some summary plots for the current data, and provides a function to view EEG waveforms
2. compute_eeg_spectra.py - calculates the alpha, beta, gamma, delta, epsilon power band across each EEG recording and each lead to generate spectral features and outputs it to eeg_spectra.csv
3. eeg_spectra_lr_pipeline.py - Runs a basic logistic regression pipeline on the spectral features and outputs ROC curve and some basic evaluation metrics.
