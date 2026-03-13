import mne
from mne.preprocessing import ICA, read_ica
import sys
import os, os.path as op
import pandas as pd
import numpy as np
import glob
from nih2mne.utilities.bids_helpers import get_mri_dict, get_project
from nih2mne.utilities.markerfile_write import main as write_markerfile
from nih2mne.utilities.trigger_utilities import (parse_marks, threshold_detect, detect_digital)
from mne.beamformer import make_lcmv, apply_lcmv, apply_lcmv_epochs, apply_lcmv_cov
import matplotlib.pyplot as plt

mne.set_config("MNE_NJOBS", "-1", set_env=True)

subject = "C10xx"
topdir = f"/data/MEGLANG/SPIN/Processed_data/sub-{subject}/ses-1/meg/"
bids_root = topdir
os.chdir(topdir)
dsets = glob.glob(f"*sub-{subject}*.ds", recursive=True)


spin1 = dsets[1]
# spin1 = f"{topdir}/sub-{subject}_SPiN_20260213_001_stitched3_raw.fif" ##LINE 38 CTF TO FIF, will cause line 71 to error
spin2 = dsets[2]
empty_room = dsets[0]
rest = dsets[3]

raw1 = mne.io.read_raw_ctf(spin1, system_clock='ignore', clean_names=True)
# raw1 = mne.io.read_raw_fif(spin1)
raw2 = mne.io.read_raw_ctf(spin2, system_clock='ignore', clean_names=True)
empty_room_raw = mne.io.read_raw_ctf(empty_room, system_clock='ignore', clean_names=True)
rest_raw = mne.io.read_raw_ctf(rest, system_clock='ignore', clean_names=True)
project = "nihmeg"
sfreq = raw1.info['sfreq']

# create variables to use within the get_mri_dict and those can be used later
# on to help set the path for where future derivatives get stored

# use os.path.join to create saving path/derivatives path
# look at nih2mne/utilities/bids_helpers.py

data_dict = get_mri_dict(subject, "/data/MEGLANG/SPIN/Processed_data/derivatives", task='SPiN', project=project, session="1")

bem = data_dict['bem'].load()
fwd = data_dict['fwd'].load()
src = data_dict['src'].load()
trans = data_dict['trans'].load()

raw1.load_data()
raw2.load_data()
empty_room_raw.load_data()
rest_raw.load_data()

raw1_filtered = raw1.copy().filter(l_freq=1.0, h_freq=None).notch_filter([60,120])
raw2_filtered = raw2.copy().filter(l_freq=1.0, h_freq=None).notch_filter([60,120])
rest_filtered = rest_raw. copy().filter(l_freq=1.0, h_freq=None).notch_filter([60,120])


regexp = r"(MLF *)"
artifact_picks = mne.pick_channels_regexp(raw1.ch_names, regexp=regexp)

dframe1 = detect_digital(spin1)
dframe2 = detect_digital(spin2)
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
    
#raw1_filtered.crop(tmin=2,tmax=632)
#raw1_filtered.annotations.append(onset=579.0, duration=4.5, description='bad_artifact')


#%% New ICA
ica1 = ICA(n_components=20, max_iter="auto", random_state=97)
ica2 = ICA(n_components=20, max_iter="auto", random_state=97)
ica_rest = ICA(n_components=20, max_iter="auto", random_state=97)

ica1.fit(raw1_filtered, reject_by_annotation=False)
ica2.fit(raw2_filtered)
ica_rest.fit(rest_filtered)



#%% ica part 1

explained_var_ratio = ica1.get_explained_variance_ratio(raw1_filtered)
for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

explained_var_ratio = ica1.get_explained_variance_ratio(
    raw1_filtered, components=[0], ch_type=["mag"]
)
# This time, print as percentage.
ratio_percent = round(100 * explained_var_ratio["mag"])
print(
    f"Fraction of variance in MEG signal explained by first component: "
    f"{ratio_percent}%"
)


ica1.plot_sources(raw1_filtered, show_scrollbars=False, title="ICA run 1") # plots ICA components in source space
ica1.plot_components() # shows ICA component heat maps in sensor space


#%% ica exclude part 1
ica1.exclude = []

# ica1.plot_overlay(raw1_filtered, exclude=ica1.exclude, picks="mag", start=0,stop=35.0)
#ica1.plot_components()

reconstruct_raw1 = raw1.copy()
ica1.apply(reconstruct_raw1)


# raw1.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
# reconstruct_raw1.plot(
#    order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False
# )

ica1.save(f'/data/MEGLANG/SPIN/Processed_data/sub-{subject}/ses-1/meg/{subject}-ica1.fif',overwrite=True)

#%%ICA part 2

explained_var_ratio = ica2.get_explained_variance_ratio(raw2_filtered)
for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

explained_var_ratio = ica2.get_explained_variance_ratio(
    raw2_filtered, components=[0], ch_type=["mag"]
)
# This time, print as percentage.
ratio_percent = round(100 * explained_var_ratio["mag"])
print(
    f"Fraction of variance in MEG signal explained by first component: "
    f"{ratio_percent}%"
)

ica2.plot_sources(raw2_filtered, show_scrollbars=False, title="ICA run 2") # plots ICA components in source space
ica2.plot_components() # shows ICA component heat maps in sensor space

#%% ica exclude part 2
ica2.exclude = []

# ica2.plot_overlay(raw2_filtered, exclude=ica2.exclude, picks="mag")
#ica2.plot_components()
reconstruct_raw2 = raw2.copy()
ica2.apply(reconstruct_raw2)

# raw2.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
# reconstruct_raw2.plot(
#     order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False
# )

ica2.save(f'/data/MEGLANG/SPIN/Processed_data/sub-{subject}/ses-1/meg/{subject}-ica2.fif', overwrite=True)

#%% rest ica

explained_var_ratio = ica_rest.get_explained_variance_ratio(rest_filtered)
for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

explained_var_ratio = ica_rest.get_explained_variance_ratio(
    raw2_filtered, components=[0], ch_type=["mag"]
)
# This time, print as percentage.
ratio_percent = round(100 * explained_var_ratio["mag"])
print(
    f"Fraction of variance in MEG signal explained by first component: "
    f"{ratio_percent}%"
)

ica_rest.plot_sources(rest_filtered, show_scrollbars=False, title="ICA rest") # plots ICA components in source space
ica_rest.plot_components() # shows ICA component heat maps in sensor space


#%% rest ica exclude

ica_rest.exclude = []

ica_rest.plot_overlay(rest_filtered, exclude=ica_rest.exclude, picks="mag", start=0.0, stop=30.0)
#ica_rest.plot_components()
reconstruct_rest = rest_raw.copy()
ica_rest.apply(reconstruct_rest)

# rest_raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
# reconstruct_rest.plot(
#     order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False
# )

ica_rest.save(f'/data/MEGLANG/SPIN/Processed_data/sub-{subject}/ses-1/meg/{subject}-ica_rest.fif',overwrite=True)

