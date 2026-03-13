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

subject = "C109"
topdir = f"/data/MEGLANG/SPIN/Processed_data/sub-{subject}/ses-1/meg/"
bids_root = topdir
os.chdir(topdir)
dsets = glob.glob(f"*sub-{subject}*.ds", recursive=True)


# spin1 = dsets[1]
spin1 = f"{topdir}/sub-C109_SPiN_20260213_001_stitched3_raw.fif" ##LINE 37 CTF TO FIF
spin2 = dsets[2]
empty_room = dsets[0]
rest = dsets[3]

# raw1 = mne.io.read_raw_ctf(spin1, system_clock='ignore', clean_names=True)
raw1 = mne.io.read_raw_fif(spin1)
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

# dframe1 = detect_digital(spin1)
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


#% Saved ICA ------- 

ica1_path = topdir + f'{subject}-ica1.fif'
ica2_path = topdir + f'{subject}-ica2.fif'
ica_rest_path = topdir + f'{subject}-ica_rest.fif'

saved_ica1 = read_ica(ica1_path)
saved_ica2 = read_ica(ica2_path)
saved_ica_rest = read_ica(ica_rest_path)

reconstruct_raw1 = raw1.copy()
saved_ica1.apply(reconstruct_raw1)

reconstruct_raw2 = raw2.copy()
saved_ica2.apply(reconstruct_raw2)

reconstruct_rest = rest_raw.copy()
saved_ica_rest.apply(reconstruct_rest)

print("ICA1 excluded:", saved_ica1.exclude)
print("ICA2 excluded:", saved_ica2.exclude)
print("ICA_rest excluded:", saved_ica_rest.exclude)

#%%-----band pass filtering
# filter for bands of interest, 0.5-3.5 delta, 4-8 theta, 8-12 alpha, 13-30 beta, 30-80 gamma 

l_freq = 30
h_freq = 80

filtered_raw1 = reconstruct_raw1.copy().filter(l_freq=l_freq, h_freq=h_freq)
filtered_raw2 = reconstruct_raw2.copy().filter(l_freq=l_freq, h_freq=h_freq)
empty_room_filtered = empty_room_raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
rest_filtered = reconstruct_rest.copy().filter(l_freq=l_freq, h_freq=h_freq)

if l_freq == 4 and h_freq == 8:
    band = "theta"
elif l_freq == 8 and h_freq == 12:
    band = "alpha"
elif l_freq == 13 and h_freq == 30:
    band = "beta"
elif l_freq >= 30:
    band = "gamma"
elif l_freq < 4:
    band = "delta"

#%# ------- epoching -------

#events = mne.find_events(reconstruct_raw, stim_channel="UPPT001")
meg_task_onset1 = raw1.annotations[0]['onset']-(1/1200)
meg_task_onset2 = raw2.annotations[0]['onset']-(1/1200)

meg_task_offset1 = raw1.annotations[10]['onset']+60
meg_task_offset2 = raw2.annotations[10]['onset']+60

cropped_raw1 = filtered_raw1.copy().crop(tmin=meg_task_onset1, tmax=meg_task_offset1)
cropped_raw2 = filtered_raw2.copy().crop(tmin=meg_task_onset2, tmax=meg_task_offset2)

meg_len = cropped_raw1.times[-1]

evt1,evt_id1=mne.events_from_annotations(cropped_raw1)
evt2,evt_id2=mne.events_from_annotations(cropped_raw2)


epochs1 = mne.Epochs(cropped_raw1, events=evt1, event_id=evt_id1, tmin=0.0, tmax=30, baseline=None, preload=True)
epochs2 = mne.Epochs(cropped_raw2, events=evt2, event_id=evt_id1, tmin=0.0, tmax=30, baseline=None, preload=True)
epochs_rest = mne.make_fixed_length_epochs(rest_filtered, duration=360, preload=True)

# epochs1.plot(events=True)
# epochs2.plot(events=True)
# epochs_rest.plot(events=True)



'''
#~~~~~~~sliding epochs

epochs = mne.make_fixed_length_epochs(reconstruct_raw, duration=60, overlap=.5, preload=False)

bands = {"4-8Hz": (4, 8), "8-12Hz": (8, 12), "30Hz": (30-60)}

sliding_spectrum = epochs.compute_psd()
full_spectrum_sliding = sliding_spectrum.plot_topomap()

epochs.plot_psd(fmin=0, fmax=60)

sliding_specific_bands = sliding_spectrum.plot_topomap(bands=bands, vlim="joint", ch_type="mag")
sliding_specific_bands
'''
#####
#% #~~~~BEAMFORMER~~~~#  with epochs use compute_covariances instead of raw


data_cov1 = mne.compute_raw_covariance(cropped_raw1, tmin=meg_task_onset1, tmax=meg_task_offset1, method="empirical")
data_cov2 = mne.compute_raw_covariance(cropped_raw2, tmin=meg_task_onset2, tmax=meg_task_offset2, method="empirical")
noise_cov = mne.compute_raw_covariance(empty_room_filtered, tmin=None, tmax=None, method='empirical')
rest_cov = mne.compute_raw_covariance(rest_filtered, tmin=None, tmax=None, method='empirical')

# data_cov1.plot(cropped_raw1.info)
# data_cov2.plot(cropped_raw2.info)
# rest_cov.plot(rest_filtered.info)

raw_rank1 = mne.compute_rank(epochs1)
raw_rank2 = mne.compute_rank(epochs2)
raw_rank_rest = mne.compute_rank(rest_filtered)

filters1 = make_lcmv(
    cropped_raw1.info,
    forward=fwd,
    data_cov=data_cov1,
    reg=0.05,
    noise_cov=noise_cov, #instead of actual data which you put in noise_cov, use empty room recording
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=raw_rank1,
)


filters2 = make_lcmv(
    cropped_raw2.info,
    forward=fwd,
    data_cov=data_cov2,
    reg=0.05,
    noise_cov=noise_cov, #instead of actual data which you put in noise_cov, use empty room recording
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=raw_rank2,
)

filters_rest = make_lcmv(
    rest_filtered.info,
    forward=fwd,
    data_cov=rest_cov,
    reg=0.05,
    noise_cov=noise_cov, #instead of actual data which you put in noise_cov, use empty room recording
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=raw_rank_rest,
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
'''


# stcs


subjects_dir = "/data/MEGLANG/SPIN/Processed_data/derivatives/freesurfer/subjects/"


# stc1 = apply_lcmv_epochs(epochs1, filters1)
# #stc_vec = apply_lcmv_epochs(epochs, filters_vec)
# stc1[0].plot(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split", title=subject+" Run 1 epoch")

# stc2 = apply_lcmv_epochs(epochs2, filters2)
# #stc_vec = apply_lcmv_epochs(epochs, filters_vec)
# stc2[0].plot(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split", title=subject+" Run 2 epoch")

# stc_rest = apply_lcmv_epochs(rest_filtered, filters_rest)
# stc_rest.plot(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split", title=subject+" Rest ")

# covariances


cov_target_only1 = mne.compute_covariance(epochs1['target_only'])
stc_target_only1 = apply_lcmv_cov(cov_target_only1, filters1)
#stc_target_only1.plot(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split", title=subject+" Run 1 target only")

cov_target_only2 = mne.compute_covariance(epochs2['target_only'])
stc_target_only2 = apply_lcmv_cov(cov_target_only2, filters2)
#stc_target_only2.plot(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split", title=subject+" Run 2 target only")

cov_one_speaker1 = mne.compute_covariance(epochs1['1_speaker'])
stc_one_speaker1 = apply_lcmv_cov(cov_one_speaker1, filters1)
#stc_one_speaker1.plot(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split", title=subject+" Run 1 1_speaker")

cov_one_speaker2 = mne.compute_covariance(epochs2['1_speaker'])
stc_one_speaker2 = apply_lcmv_cov(cov_one_speaker2, filters2)
#stc_one_speaker2.plot(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split", title=subject+" Run 2 1_speaker")

cov_two_speaker1 = mne.compute_covariance(epochs1['2_speaker'])
stc_two_speaker1 = apply_lcmv_cov(cov_two_speaker1, filters1)
#stc_two_speaker1.plot(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split", title=subject+" Run 1 2_speaker")

cov_two_speaker2 = mne.compute_covariance(epochs2['2_speaker'])
stc_two_speaker2 = apply_lcmv_cov(cov_two_speaker2, filters2)
#stc_two_speaker2.plot(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split", title=subject+" Run 2 2_speaker")

cov_SSN1 = mne.compute_covariance(epochs1['SSN'])
stc_SSN1 = apply_lcmv_cov(cov_SSN1, filters1)
#stc_SSN1.plot(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split", title=subject+" Run 1 SSN")

cov_SSN2 = mne.compute_covariance(epochs2['SSN'])
stc_SSN2 = apply_lcmv_cov(cov_SSN2, filters2)
#stc_SSN2.plot(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split", title=subject+" Run 2 SSN")

cov_rest = mne.compute_covariance(epochs_rest)
stc_rest = apply_lcmv_cov(cov_rest, filters_rest)
# stc_rest.plot(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split", title=subject+" Rest")

# contrasts
plot_options1 = dict(subject="sub-"+subject, subjects_dir=subjects_dir, hemi="split")
import copy

##Run 1

stc_diff1 = copy.deepcopy(stc_target_only1)
stc_diff1._data = stc_one_speaker1._data - stc_target_only1._data
#stc_diff1.plot(**plot_options1, title = subject + " Run 1: one - target")

stc_diff2 = copy.deepcopy(stc_two_speaker1)
stc_diff2._data = stc_two_speaker1._data - stc_target_only1._data
#stc_diff2.plot(**plot_options1, title = subject + " Run 1: two - target")

stc_diff3 = copy.deepcopy(stc_SSN1)
stc_diff3._data = stc_SSN1._data - stc_target_only1._data
#stc_diff3.plot(**plot_options1, title = subject + " Run 1: SSN - target")

##Run 2

stc_diff4 = copy.deepcopy(stc_target_only2)
stc_diff4._data = stc_one_speaker2._data - stc_target_only2._data
#stc_diff4.plot(**plot_options1, title = subject + " Run 2: one - target")

stc_diff5 = copy.deepcopy(stc_two_speaker2)
stc_diff5._data = stc_two_speaker2._data - stc_target_only2._data
#stc_diff5.plot(**plot_options1, title = subject + " Run 2: two - target")

stc_diff6 = copy.deepcopy(stc_SSN2)
stc_diff6._data = stc_SSN2._data - stc_target_only2._data
#stc_diff6.plot(**plot_options1, title = subject + " Run 2: SSN - target")


##Rest

stc_diff7 = copy.deepcopy(stc_rest)
stc_diff7._data = stc_target_only1._data - stc_rest._data
#stc_diff7.plot(**plot_options1, title = subject + " Run 1: Target - Rest ")

stc_diff8 = copy.deepcopy(stc_rest)
stc_diff8._data = stc_target_only2._data - stc_rest._data
#stc_diff8.plot(**plot_options1, title = subject + " Run 2: Target - Rest ")

## Difference after both runs combined

stc_add1 = copy.deepcopy(stc_one_speaker1)
stc_add1._data = (stc_one_speaker1._data + stc_one_speaker2._data) - (stc_target_only1._data + stc_target_only2._data)
# stc_add1.plot(**plot_options1, title = subject + " 1_speaker - target " + band)

stc_add2 = copy.deepcopy(stc_two_speaker1)
stc_add2._data = (stc_two_speaker1._data + stc_two_speaker2._data) - (stc_target_only1._data + stc_target_only2._data)
# stc_add2.plot(**plot_options1, title = subject + " 2_speaker - target " + band)

stc_add3 = copy.deepcopy(stc_SSN1)
stc_add3._data = (stc_SSN1._data + stc_SSN2._data) - (stc_target_only1._data + stc_target_only2._data)
# stc_add3.plot(**plot_options1, title = subject + " SSN - target " + band)

stc_add4= copy.deepcopy(stc_target_only1)
stc_add4._data = (stc_target_only1._data + stc_target_only2._data) - (stc_rest._data)
# stc_add4.plot(**plot_options1, title = subject + " Target - rest " + band)

#% save the averages

morph1 = mne.compute_source_morph(stc_add1, subject_from="sub-"+subject, subject_to="fsaverage", subjects_dir=subjects_dir)
morph2 = mne.compute_source_morph(stc_add2, subject_from="sub-"+subject, subject_to="fsaverage", subjects_dir=subjects_dir)
morph3 = mne.compute_source_morph(stc_add3, subject_from="sub-"+subject, subject_to="fsaverage", subjects_dir=subjects_dir)
morph4 = mne.compute_source_morph(stc_add4, subject_from="sub-"+subject, subject_to="fsaverage", subjects_dir=subjects_dir)

stc_fsavg1 = morph1.apply(stc_add1)
stc_fsavg2 = morph2.apply(stc_add2)
stc_fsavg3 = morph3.apply(stc_add3)
stc_fsavg4 = morph4.apply(stc_add4)

stc_fsavg1.save(f"/data/MEGLANG/SPIN/Processed_data/derivatives/nihmeg/sub-{subject}/ses-1/meg/{subject}_1speaker_contrast_{band}", overwrite=True)
stc_fsavg2.save(f"/data/MEGLANG/SPIN/Processed_data/derivatives/nihmeg/sub-{subject}/ses-1/meg/{subject}_2speaker_contrast_{band}", overwrite=True)
stc_fsavg3.save(f"/data/MEGLANG/SPIN/Processed_data/derivatives/nihmeg/sub-{subject}/ses-1/meg/{subject}_SSN_contrast_{band}", overwrite=True)
stc_fsavg4.save(f"/data/MEGLANG/SPIN/Processed_data/derivatives/nihmeg/sub-{subject}/ses-1/meg/{subject}_target_only_contrast_{band}", overwrite=True)
