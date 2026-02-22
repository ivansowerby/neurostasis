import mne
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from data_loader import resting_state_ses_1_eyesopen_loader, resting_state_ses_1_eyesclosed_loader, resting_state_ses_2_eyesopen_loader, resting_state_ses_2_eyesclosed_loader
from utils import plot_eeg, plot_spectrum

# %%

def bandpass(data, bp_lo=0.5, bp_hi=50, fs=500, order=4):
    nyq = 0.5 * fs
    b, a = sps.butter(order, [bp_lo/nyq, bp_hi/nyq], btype='band')
    return sps.filtfilt(b, a, data, axis=1)

def notchpass(data, freq=50, fs=500, Q=30):
    b, a = sps.iirnotch(freq, Q, fs)
    return sps.filtfilt(b, a, data, axis=1)

def cleaning(data, bp_lo=0.5, bp_hi=50, bp_order=4, notch_freq=50, fs=500, Q=30):
    # Demean
    data = data - np.mean(data, keepdims=True)
    
    # Bandpass
    data = bandpass(data, bp_lo, bp_hi, fs, bp_order)
    
    # Notch pass
    data = notchpass(data, notch_freq, fs, Q)
    
    # Common Average Referencing
    data = data - np.mean(data, keepdims=True)
    
def run_ica_four_dicts(
    s1_ec, s1_eo,
    s2_ec, s2_eo,
    ch_names,
    fs=500,
    n_components=0.99
):
    """
    Fit ICA per subject using all sessions/conditions,
    then apply consistently to all four dictionaries.
    """

    cleaned_s1_ec = {}
    cleaned_s1_eo = {}
    cleaned_s2_ec = {}
    cleaned_s2_eo = {}

    subjects = s1_ec.keys()

    for sub in subjects:
        print(f"\nRunning ICA for subject {sub}")

        # -------------------------
        # Concatenate all data for that subject
        # -------------------------
        all_data = np.concatenate([
            s1_ec[sub],
            s1_eo[sub],
            s2_ec[sub],
            s2_eo[sub]
        ], axis=1)

        # Create Raw object
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=fs,
            ch_types=['eeg'] * len(ch_names)
        )

        raw = mne.io.RawArray(all_data, info)
        raw.set_eeg_reference('average', projection=False)

        # ICA training copy (high-pass 1 Hz)
        raw_for_ica = raw.copy().filter(l_freq=1., h_freq=None)

        # Fit ICA
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method='fastica',
            random_state=42,
            max_iter='auto'
        )

        ica.fit(raw_for_ica)

        # Automatic artifact detection
        ica.exclude = []

        ica.exclude = list(set(ica.exclude))

        print("Excluded components:", ica.exclude)

        # -------------------------
        # Apply ICA separately to each dataset
        # -------------------------
        def apply_ica(data):
            raw_tmp = mne.io.RawArray(data, info)
            raw_tmp.set_eeg_reference('average', projection=False)
            raw_clean = ica.apply(raw_tmp.copy())
            return raw_clean.get_data()

        cleaned_s1_ec[sub] = apply_ica(s1_ec[sub])
        cleaned_s1_eo[sub] = apply_ica(s1_eo[sub])
        cleaned_s2_ec[sub] = apply_ica(s2_ec[sub])
        cleaned_s2_eo[sub] = apply_ica(s2_eo[sub])

    return cleaned_s1_ec, cleaned_s1_eo, cleaned_s2_ec, cleaned_s2_eo


n_channels = 61
n_subjects = 28
dtype = np.float32
fs = 500
base_path = "data/resting_state"
ch_names = np.loadtxt('data/resting_state/sub-01/ses-1/eeg/sub-01_ses-1_electrodes.tsv', delimiter='\t', usecols=0, skiprows=1, dtype=str).tolist()

s1_eo = resting_state_ses_1_eyesopen_loader(base_path, n_channels, n_subjects, dtype)

s1_ec = resting_state_ses_1_eyesclosed_loader(base_path, n_channels, n_subjects, dtype)

s2_ec = resting_state_ses_2_eyesclosed_loader(base_path, n_channels, n_subjects, dtype)

s2_eo = resting_state_ses_2_eyesopen_loader(base_path, n_channels, n_subjects, dtype)

s1_ec_clean, s1_eo_clean, s2_ec_clean, s2_eo_clean = run_ica_four_dicts(s1_ec, s1_eo, s2_ec, s2_eo, ch_names)

# %%