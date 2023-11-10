import os
import numpy as np
import pandas as pd
import mne
import json
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def view_eeg(datafolder, pt_id, offset, time_length):
    """
    Generates a plot to visualise the pre-processed EEG from the ds004504 dataset of a single patient

    Args:
        datafolder (string) : path to the top directory where the dataset is located
        pt_id (string) : the id of the patient
        offset (float) : Time in seconds of the point you want to start visualising
        time_length (float) : Time in seconds of the length of the recording you want to view from offset

    Returns:
        None

    """
    # Construct paths for the recording details and recording data
    raw_path = "/".join((datafolder, pt_id, "eeg"))
    processed_path = "/".join((datafolder, "derivatives", pt_id, "eeg"))
    if not os.path.exists(raw_path) or not os.path.exists(processed_path):
        print("Error in path to data")
        return
    # Get the file name for the recording details
    recording_fn = [x for x in os.listdir(raw_path) if x.find("json") > -1]
    if len(recording_fn) == 0:
        print("No recording file found")
        return
    recording_fn = raw_path + "/" + recording_fn[0]
    with open(recording_fn, "r") as fn:
        recording_details = json.load(fn)
    hz = recording_details["SamplingFrequency"]
    duration = recording_details["RecordingDuration"]
    if offset > duration:
        print("Recording is only {} seconds long, offset is too large".format(duration))
        return
    eeg_fn = "/".join((processed_path, "{}_task-eyesclosed_eeg.set".format(pt_id)))
    if not os.path.exists(eeg_fn):
        print("Recording doesn't exist")
        return
    channels = pd.read_csv("/".join((raw_path, "{}_task-eyesclosed_channels.tsv".format(pt_id))), sep="\t")["name"].to_list()
    eeg = sio.loadmat(eeg_fn)["data"]
    start = offset * hz
    end = (offset + time_length) * hz
    end = end if end < eeg.shape[1] else eeg.shape[1] - 1
    eeg = eeg[:, start:end]
    fig, ax = plt.subplots(len(channels) // 2 + 1, 2)
    for i, c in enumerate(channels):
        ax[i // 2, i % 2].set_title(c, fontsize=8)
        ax[i // 2, i % 2].set_xlabel("Time [s]", fontsize=8)
        ax[i // 2, i % 2].set_ylabel("microV", fontsize=8)
        ax[i // 2, i % 2].plot(np.linspace(offset, offset + time_length, eeg.shape[1]), eeg[i, :])
    


dataset = "openneuro_ds004504"
pt = "sub-001"
idx = pd.read_csv(dataset + "/participants.tsv", sep="\t")

# Plot MMSE distribution by dementia status
idx.groupby("Group")["MMSE"].plot.hist(title="Distribution of MMSE scores by dementia status", xlabel="MMSE")
plt.legend(["Alzheimer's", "Healthy", "Frontotemporal dementia"])
plt.axvline(25, color="k")
plt.text(25, 15, "Normal MMSE threshold")
plt.savefig("mmsedist.png")
plt.close()

# Plot dementia and gender distribution
idx.groupby(["Gender", "Group"]).size().unstack().plot.bar(stacked=True, 
                                                           title="Proportion of dementia status by gender", ylabel="counts")
plt.legend(["Alzheimer's", "Healthy", "Frontotemporal dementia"])
plt.savefig("agedementiadist.png")
plt.close()

# Plot example eeg
view_eeg(dataset, "sub-001", 0, 10)
plt.savefig("sub-001_example_eeg.png")
plt.close()

# PCA of eeg spectra
spectra = pd.read_csv("eeg_spectra.csv")
spectra_pca = PCA(2)
spectra_pca.fit(spectra.iloc[:, 5:])
group_map = {"A" : "Alzheimer's", "C" : "Healthy", "F" : "Frontotemporal"}
variance = spectra_pca.explained_variance_ratio_
xs = spectra_pca.transform(spectra.iloc[:, 5:])
for g in spectra["Group"].unique():
    plt.scatter(xs[spectra["Group"] == g, 0], xs[spectra["Group"] == g, 1], label=group_map[g])
plt.xlabel("PC1 ({:.2f}% variance explained)".format(variance[0] * 100))
plt.ylabel("PC2 ({:.2f}% variance explained)".format(variance[1] * 100))
plt.legend()
plt.title("PCA of classical eeg bands across all leads")
plt.savefig("eeg_spectra_pca.png")
plt.close()