import gpype as gp 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Find the most recent CSV file from g.Pype recordings
csv_files = glob.glob("Data/gTec/testdata_eeg_20260221_215823*.csv")
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
for i, ch in enumerate(channels[:8]):
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

# %%

fs = 250
dt = 1/fs
N = len(time)
window = sps.windows.hann(N)
for i, ch in enumerate(channels[:8]):
    freqs = sp.fft.rfftfreq(N, d=dt)[1:]
    spec = sp.fft.rfft(data[ch]*window)[1:]
    mags = np.abs(spec)
    plt.plot(freqs, mags, label=f'Ch {i+1}')
    
plt.legend()
plt.show()
# %%

if __name__ == "__main__":

    # Create the main application window
    app = gp.MainApp()

    # Create processing pipeline
    p = gp.Pipeline()

    # Generate 10 Hz rectangular wave (rich in harmonics)
    source = channels[:8]

    # FFT analysis with windowing (1 second windows, 50% overlap)
    fft = gp.FFT(
        window_size=fs,  # 250 samples = 1 sec window
        overlap=0.5,  # 50% overlap for smooth updates
        window_function="hamming",
    )  # Reduce spectral leakage

    # Frequency domain visualization (spectrum analyzer)
    scope = gp.SpectrumScope(amplitude_limit=20)  # Y-axis: 0-20 dB

    # Connect processing chain: source -> FFT -> spectrum display
    p.connect(source, fft)
    p.connect(fft, scope)

    # Add spectrum scope to application window
    app.add_widget(scope)

    # Start pipeline and run application
    p.start()  # Begin signal processing
    app.run()  # Show GUI and start main loop
    p.stop()  # Clean shutdown