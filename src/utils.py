import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spf
import scipy.signal as sps
import pandas as pd
import glob
import os

def plot_eeg(file_path, num_channels=8): # file path and no. of channels to plot
    # Find the most recent CSV file from g.Pype recordings
    csv_files = glob.glob(file_path) # file_path is string of the form "file_path*.csv"
    if not csv_files:
        raise FileNotFoundError("No CSV files found.")
    file_path = max(csv_files, key=os.path.getmtime)  # Most recent file

    # Load recorded data into pandas DataFrame
    data = pd.read_csv(file_path)

    # Extract time index and channel data
    time = data["Time"]  # Sample timestamps
    channels = data.columns[1:]  # All data columns (signals + events)

    # Create multi-channel EEG-style plot
    plt.figure(figsize=(10, 6))

    # Channel stacking parameters for clear visualization
    offset = -100  # Vertical spacing between channels
    yticks = []  # Y-axis tick positions
    yticklabels = []  # Y-axis tick labels

    # Plot each channel with vertical offset
    adj_data = []
    for i, ch in enumerate(channels[:num_channels+1]):
        channel_offset = i * offset
        adj_data.append(data[ch] + channel_offset) 
        plt.plot(time, adj_data[i], label=ch)
        yticks.append(channel_offset)
        yticklabels.append(f"Ch{i + 1}")

    # Configure plot appearance
    plt.yticks(yticks, yticklabels)
    plt.xlabel("Time (s)")
    plt.title("EEG Recordings")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.ylim((len(channels)) * offset, -offset)

    # Display the plot
    plt.tight_layout()
    plt.show()
    
    return time, channels, adj_data

def plot_spectrum(data, time, fs):
    dt = 1/fs
    N = len(time)
    window = sps.windows.hann(N)
    freqs = spf.rfftfreq(N, d=dt)[1:]
    spec = spf.rfft(data*window)[1:]
    mags = np.abs(spec)
    plt.plot(freqs, mags) 
    plt.legend()
    plt.show()
    return freqs, spec, mags

