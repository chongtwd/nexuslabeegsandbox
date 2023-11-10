import numpy as np
import pandas as pd
import scipy.io as sio
import json
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import simps
import itertools

def load_eeg(stem, pt_id):
    eeg_fn = "/".join((stem, "derivatives", pt_id, "eeg", "{}_task-eyesclosed_eeg.set".format(pt_id)))
    channels_fn = "/".join((stem, pt_id, "eeg", "{}_task-eyesclosed_channels.tsv".format(pt_id)))
    channels = pd.read_csv(channels_fn, sep="\t")["name"].to_list()
    with open("/".join((stem, pt_id, "eeg", "{}_task-eyesclosed_eeg.json".format(pt_id)))) as fn:
        fs = json.load(fn)["SamplingFrequency"]
    eeg = sio.loadmat(eeg_fn)["data"]
    return fs, pd.DataFrame(eeg.T, columns=channels)

def summariseEEGSpectrum(x, fs, winSize=500):
    # Summarise EEG spectrum by classical bands
    out = {}
    for i in range(x.shape[1]):
        lead = {}
        freqs, psd = welch(x.iloc[:,i], fs=fs, nperseg=winSize)
        total = simps(psd, dx=freqs[1]-freqs[0])
        lead["alpha"] = simps(psd[(freqs >= 8) & (freqs < 14)], dx=freqs[1]-freqs[0]) / total
        lead["beta"] = simps(psd[(freqs >= 14) & (freqs < 30)], dx=freqs[1]-freqs[0]) / total
        lead["gamma"] = simps(psd[(freqs >= 30)], dx=freqs[1]-freqs[0]) / total
        lead["delta"] = simps(psd[(freqs >= 0.5) & (freqs < 4)], dx=freqs[1]-freqs[0]) / total
        lead["theta"] = simps(psd[(freqs >= 4) & (freqs < 8)], dx=freqs[1]-freqs[0]) / total
        out[x.columns[i]] = lead
    return pd.DataFrame(out).T.reset_index().rename({"index" : "Lead"}, axis=1)

dataset = "openneuro_ds004504"
idx = pd.read_csv(dataset + "/participants.tsv", sep="\t")
spectra = []
for pt in idx["participant_id"]:
    print("Processing {}".format(pt))
    fs, data = load_eeg(dataset, pt)
    df = summariseEEGSpectrum(data, fs)
    newcols = ["_".join(x) for x in itertools.product(df["Lead"], df.columns[1:])]
    df = pd.DataFrame(df.iloc[:, 1:].to_numpy().reshape(1, -1), columns=newcols)
    df["participant_id"] = pt
    spectra.append(df)
spectra = pd.concat(spectra, axis=0)
combined = idx.merge(spectra, on="participant_id")
combined.to_csv("eeg_spectra.csv", index=False)