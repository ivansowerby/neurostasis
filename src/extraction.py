import gpype as gp 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_eeg, plot_spectrum
from mne.time_frequency import psd_array_welch
from scipy.stats import entropy
from scipy.stats import linregress

# %%
# Load the pre-processed data
data = np.load('data/cleaned_resting_state_data.npz', allow_pickle=True)
s1_ec, s1_eo, s2_ec, s2_eo = data['arr_0'].item(), data['arr_1'].item(), data['arr_2'].item(), data['arr_3'].item()

# %%
# Plotting functions
def do_psd_plots(data, eyes, session_num, fs=500):
    fig, ax = plt.subplots(1, 4, figsize=(10, 6))
    ax = ax.flatten()
    for i in range(1, 29):
        session = {
            "S"+session_num+" "+eyes: s1_ec[f'sub-{i:02d}']
            }
        
    
        for label, data in session.items():
            psd, freqs = psd_array_welch(
                data,
                sfreq=fs,
                fmin=0.5,
                fmax=45,
                n_fft=1024,
                n_overlap=512
                )
            psd_mean = np.mean(psd, axis=0)
            
            ax[i-1].plot(freqs, 10 * np.log10(psd_mean), label=label)
            
        ax[i-1].set_xlabel('Frequency (Hz)')
        ax[i-1].set_ylabel('Power (dB)')
        ax[i-1].set_title(f'PSD Across Sessions — Subject {i},' + eyes)

    ax[0].legend()

    plt.show()

def plot_all_subject_psds(
    s1_ec,
    s1_eo,
    s2_ec,
    s2_eo,
    fs=500,
    fmin=0.5,
    fmax=45
):
    subjects = list(s1_ec.keys())
    
    fig, axes = plt.subplots(7, 4, figsize=(22, 28))
    axes = axes.flatten()

    for i, sub in enumerate(subjects[:3]):

        ax = axes[i]

        sessions = {
            'S1 EC': s1_ec[sub],
            #'S1 EO': s1_eo[sub],
            'S2 EC': s2_ec[sub]#,
            #'S2 EO': s2_eo[sub]
        }

        for label, data in sessions.items():

            psd, freqs = psd_array_welch(
                data,
                sfreq=fs,
                fmin=fmin,
                fmax=fmax,
                n_fft=1024,
                n_overlap=512,
                verbose=False
            )


            psd_mean = np.mean(psd, axis=0)

            ax.plot(freqs, 10 * np.log10(psd_mean), label=label)

        ax.set_title(sub, fontsize=9)
        ax.set_xlim(fmin, fmax)
        ax.tick_params(labelsize=7)

        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

def do_per_subject_plots(
        s1_ec,
        s1_eo,
        s2_ec,
        s2_eo,
        fs=500,
        fmin=0.5,
        fmax=45
):
    subjects = list(s1_ec.keys())
    for sub in subjects:
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
        
        sessions = [
            ("S1 Eyes Closed", s1_ec[sub]),
            ("S1 Eyes Open",  s1_eo[sub]),
            ("S2 Eyes Closed", s2_ec[sub]),
            ("S2 Eyes Open",  s2_eo[sub])
            ]
        
        for ax, (label, data) in zip(axes, sessions):
            
            psd, freqs = psd_array_welch(
                data,
                sfreq=fs,
                fmin=0.5,
                fmax=45,
                n_fft=1024,
                n_overlap=512,
                verbose=False
                )
            
            psd_mean = np.mean(psd, axis=0)
            
            ax.plot(freqs, 10 * np.log10(psd_mean))
            ax.set_title(label, fontsize=10)
            ax.set_xlabel('Frequency/Hz')
            ax.set_ylabel('Amplitude')
            ax.set_xlim(0, 45)
            ax.grid(alpha=0.3)
            
        axes[0].set_ylabel("Power (dB)")
        fig.suptitle(f"PSD — Subject {sub}", fontsize=14)
            
        plt.tight_layout()
        plt.show()

def extract_psd_features(data, fs=500,
                         fmin=1, fmax=45,
                         theta_band=(4, 8),
                         alpha_band=(8, 12)):
    
    # Compute PSD 
    freqs, psd = sps.welch(
        data,
        fs=fs,
        nperseg=fs*2,
        noverlap=fs,
        axis=1
    )

    # Keep frequency range
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    psd = psd[:, mask]

    # Average across channels
    psd_mean = np.mean(psd, axis=0)

    # Alpha and theta band powers
    theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])

    theta_power = np.trapezoid(psd_mean[theta_mask], freqs[theta_mask])
    alpha_power = np.trapezoid(psd_mean[alpha_mask], freqs[alpha_mask])
    theta_alpha_ratio = theta_power / alpha_power if alpha_power != 0 else np.nan

    # Peak alpha frequency
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = psd_mean[alpha_mask]
    peak_alpha_freq = alpha_freqs[np.argmax(alpha_psd)]

    # Spectral slope (1/f)
    log_freqs = np.log10(freqs)
    log_psd = np.log10(psd_mean)
    slope, _, _, _, _ = linregress(log_freqs, log_psd)

    # Spectral entropy
    psd_norm = psd_mean / np.sum(psd_mean)
    spec_entropy = entropy(psd_norm)

    return {
        "alpha_power": alpha_power,
        "theta_power": theta_power,
        "theta_alpha_ratio": theta_alpha_ratio,
        "spectral_slope": slope,
        "peak_alpha_frequency": peak_alpha_freq,
        "spectral_entropy": spec_entropy
    }

def extract_all_subjects(s1_ec,
                         s1_eo,
                         s2_ec,
                         s2_eo,
                         fs=500):

    results = []

    subjects = s1_ec.keys()

    for sub in subjects:

        for condition, d1, d2 in [
            ("EC", s1_ec, s2_ec),
            ("EO", s1_eo, s2_eo)
        ]:

            features_s1 = extract_psd_features(d1[sub], fs)
            features_s2 = extract_psd_features(d2[sub], fs)

            row = {
                "subject": sub,
                "condition": condition,

                # session 1
                "alpha_s1": features_s1["alpha_power"],
                "theta_s1": features_s1["theta_power"],
                "ratio_s1": features_s1["theta_alpha_ratio"],
                "slope_s1": features_s1["spectral_slope"],
                "paf_s1": features_s1["peak_alpha_frequency"],
                "entropy_s1": features_s1["spectral_entropy"],

                # session 2
                "alpha_s2": features_s2["alpha_power"],
                "theta_s2": features_s2["theta_power"],
                "ratio_s2": features_s2["theta_alpha_ratio"],
                "slope_s2": features_s2["spectral_slope"],
                "paf_s2": features_s2["peak_alpha_frequency"],
                "entropy_s2": features_s2["spectral_entropy"],
            }

            results.append(row)

    return pd.DataFrame(results)

# %%
# plot different types of graphs
s1_ec_freqs, s1_ec_psds = do_psd_plots(s1_ec, 'eyes closed', str(1))
s1_eo_freqs, s1_eo_psds = do_psd_plots(s1_eo, 'eyes open', str(1))
s2_ec_freqs, s2_ec_psds = do_psd_plots(s2_ec, 'eyes closed', str(2))
s2_eo_freqs, s2_eo_psds = do_psd_plots(s2_eo, 'eyes open', str(2))
# %%
plot_all_subject_psds(s1_ec, s1_eo, s2_ec, s2_eo)
# %%
do_per_subject_plots(s1_ec, s1_eo, s2_ec, s2_eo)
# %%
"""
feature matrix of alpha power, theta power, theta/alhpa ratio, spectral slope,
peak alpha frequency (paf), spectral entropy
"""
features = extract_all_subjects(s1_ec, s1_eo, s2_ec, s2_eo)

features.to_csv("eeg_psd_features.csv", index=False)