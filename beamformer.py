import mne
from mne.preprocessing import ICA
import sys
import os, os.path as op
import pandas as pd
import numpy as np
import glob
from nih2mne.utilities.bids_helpers import get_mri_dict, get_project
from nih2mne.utilities.markerfile_write import main as write_markerfile
from nih2mne.utilities.trigger_utilities import (parse_marks, threshold_detect, detect_digital)
from mne.beamformer import make_lcmv, apply_lcmv, apply_lcmv_epochs
import matplotlib.pyplot as plt


subject = "C101"
topdir = f"/data/MEGLANG/SPIN/Processed_data/sub-{subject}/ses-1/meg/"
bids_root = topdir
os.chdir(topdir)
dsets = glob.glob(f"*sub-{subject}*.ds", recursive=True)
fname = dsets[1]
raw = mne.io.read_raw_ctf(fname,system_clock='ignore', clean_names=True)
project = "nihmeg"

# create variables to use within the get_mri_dict and those can be used later
# on to help set the path for where future derivatives get stored

# use os.path.join to create saving path/derivatives path
# look at nih2mne/utilities/bids_helpers.py

data_dict = get_mri_dict(subject, "/data/MEGLANG/SPIN/Processed_data/derivatives", task='SPiN', project=project, session="1")

bem = data_dict['bem'].load()
fwd = data_dict['fwd'].load()
src = data_dict['src'].load()
trans = data_dict['trans'].load()

raw.load_data()

raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=None)
evt,evt_id=mne.events_from_annotations(raw)

#raw.plot()

regexp = r"(MLF *)"
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)

dframe = detect_digital(fname)
'''
auditory_delay = 0.048
dframe.onset += auditory_delay   #Add the auditory delay

condition_dict = {
                  'target_only': 8,    #<<<Fill these in with your condition naming
                  '1_speaker': 16,
                  'SSN': 32,
                  '2_speaker': 64
                  }

#Rename your events
for key,value in condition_dict.items():
    dframe.loc[dframe.condition==key, 'condition']=value

try:
    write_markerfile(dframe, fname)
    print(f'Wrote markers to {fname}')
except:
    print(f'Failed to write markerfile for {fname}')

    '''
# ------- ICA ------- 
ica = ICA(n_components=20, max_iter="auto", random_state=97)
ica.fit(raw_filtered)
#ica

explained_var_ratio = ica.get_explained_variance_ratio(raw_filtered)
for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

explained_var_ratio = ica.get_explained_variance_ratio(
    raw_filtered, components=[0], ch_type=["mag"]
)
# This time, print as percentage.
ratio_percent = round(100 * explained_var_ratio["mag"])
print(
    f"Fraction of variance in MEG signal explained by first component: "
    f"{ratio_percent}%"
)


ica.plot_sources(raw_filtered, show_scrollbars=False) # plots ICA components in source space
ica.plot_components() # shows ICA component heat maps in sensor space

ica.exclude = [0,1,2,3,4,5,10]

ica.plot_overlay(raw_filtered, exclude=ica.exclude, picks="mag")
ica.plot_components()
reconstruct_raw = raw.copy()
ica.apply(reconstruct_raw)

raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
reconstruct_raw.plot(
    order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False
)

# filter for bands of interest

filtered_raw = reconstruct_raw.copy().filter(l_freq=4.0, h_freq=8)
#reconstruct_raw.plot()

# ------- epoching -------

events = mne.find_events(reconstruct_raw, stim_channel="UPPT001")

#events_cleaned = events[:-1] # instead use raw.crop to crop time off first and last epoch
cropped_raw = filtered_raw.copy().crop(tmin=15, tmax=615)

epochs = mne.Epochs(cropped_raw, events=events, event_id=evt,evt_id, tmin=0.0, tmax=.0, baseline=(0,0), preload=True)

epochs.plot(n_epochs=4, events=True)

target_only_epoch = epochs["target_only"]
one_speaker_epoch = epochs["1_speaker"]
two_speaker_epoch = epochs["2_speaker"]
SSN_epoch = epochs["SSN"]


spectrum = epochs["target_only"].compute_psd() #adjust for condition, or figure out loop
full_spectrum_plot = spectrum.plot_topomap()

epochs["target_only"].plot_psd(fmin=0,fmax=60)

#bands = {"4-8Hz": (4, 8), "8-12Hz": (8, 12), "30Hz": 30}
#specific_bands = spectrum.plot_topomap(bands=bands, vlim="joint", ch_type="mag")

epochs["target_only"].plot_image(picks="mag", combine="mean")
'''

#~~~~~~~sliding epochs
'''
epochs = mne.make_fixed_length_epochs(reconstruct_raw, duration=60, overlap=.5, preload=False)

#bands = {"4-8Hz": (4, 8), "8-12Hz": (8, 12), "30Hz": 30}

sliding_spectrum = epochs.compute_psd()
full_spectrum_sliding = sliding_spectrum.plot_topomap()

epochs.plot_psd(fmin=0, fmax=60)

sliding_specific_bands = sliding_spectrum.plot_topomap(bands=bands, vlim="joint", ch_type="mag")
sliding_specific_bands

##

'''
#~~~~BEAMFORMER~~~~#  with epochs use compute_covariances instead of raw
data_cov = mne.compute_raw_covariance(cropped_raw, tmin=0.0, tmax=600.0, method="empirical")
noise_cov = mne.compute_raw_covariance(cropped_raw, tmin=0, tmax=600.0, method='empirical')
data_cov.plot(cropped_raw.info)

filters = make_lcmv(
    cropped_raw.info,
    forward=fwd,
    data_cov=data_cov,
    reg=0.05,
    noise_cov=None, #instead of actual data which you put in noise_cov, use empty room recording
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None,
)
'''
filters_vec = make_lcmv(
    cropped_raw.info,
    forward=fwd,
    data_cov=data_cov,
    reg=0.05,
    noise_cov=None,
    pick_ori="vector",
    weight_norm="unit-noise-gain-invariant",
    rank=None,
)



stc = apply_lcmv_epochs(epochs, filters)
stc_vec = apply_lcmv_epochs(epochs, filters_vec)
