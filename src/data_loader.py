import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def resting_state_ses_1_eyesopen_loader(base_path, n_channels, n_subjects, dtype):
    session_data = {}

    for sub in range(1, n_subjects + 1):
        
        subject_id = f"sub-{sub:02d}"
        
        file_path = os.path.join(
            base_path,
            subject_id,
            "ses-1",
            "eeg",
            f"{subject_id}_ses-1_task-eyesopen_eeg.fdt"
            )
        
        if os.path.exists(file_path):
            
            data = np.fromfile(file_path, dtype=dtype)
            
            n_samples = len(data) // 61
            usable_values = n_samples * 61

            data = data[:usable_values]  # trim extra values
            data = data.reshape((n_samples, 61)).T
            
            session_data[subject_id] = data
            
            print(f"{subject_id} loaded: {data.shape}")
            
        else:
                print(f"{subject_id} missing file")
                
    return session_data


def resting_state_ses_1_eyesclosed_loader(base_path, n_channels, n_subjects, dtype):
    session_data = {}

    for sub in range(1, n_subjects + 1):
        
        subject_id = f"sub-{sub:02d}"
        
        file_path = os.path.join(
            base_path,
            subject_id,
            "ses-1",
            "eeg",
            f"{subject_id}_ses-1_task-eyesclosed_eeg.fdt"
            )
        
        if os.path.exists(file_path):
            
            data = np.fromfile(file_path, dtype=dtype)
            
            n_samples = len(data) // 61
            usable_values = n_samples * 61

            data = data[:usable_values]  # trim extra values
            data = data.reshape((n_samples, 61)).T
            
            session_data[subject_id] = data
            
            print(f"{subject_id} loaded: {data.shape}")
            
        else:
                print(f"{subject_id} missing file")
                
    return session_data

def resting_state_ses_2_eyesopen_loader(base_path, n_channels, n_subjects, dtype):
    session_data = {}

    for sub in range(1, n_subjects + 1):
        
        subject_id = f"sub-{sub:02d}"
        
        file_path = os.path.join(
            base_path,
            subject_id,
            "ses-2",
            "eeg",
            f"{subject_id}_ses-2_task-eyesopen_eeg.fdt"
            )
        
        if os.path.exists(file_path):
            
            data = np.fromfile(file_path, dtype=dtype)
            
            n_samples = len(data) // 61
            usable_values = n_samples * 61

            data = data[:usable_values]  # trim extra values
            data = data.reshape((n_samples, 61)).T
            session_data[subject_id] = data
            
            print(f"{subject_id} loaded: {data.shape}")
            
        else:
                print(f"{subject_id} missing file")
                
    return session_data

def resting_state_ses_2_eyesclosed_loader(base_path, n_channels, n_subjects, dtype):
    session_data = {}

    for sub in range(1, n_subjects + 1):
        
        subject_id = f"sub-{sub:02d}"
        
        file_path = os.path.join(
            base_path,
            subject_id,
            "ses-2",
            "eeg",
            f"{subject_id}_ses-2_task-eyesclosed_eeg.fdt"
            )
        
        if os.path.exists(file_path):
            
            data = np.fromfile(file_path, dtype=dtype)
            
            n_samples = len(data) // 61
            usable_values = n_samples * 61

            data = data[:usable_values]  # trim extra values
            data = data.reshape((n_samples, 61)).T
            
            session_data[subject_id] = data
            
            print(f"{subject_id} loaded: {data.shape}")
            
        else:
                print(f"{subject_id} missing file")
                
    return session_data

def gtec_loader(file_path): # exclude .csv
    csv_files = glob.glob(f"{file_path}*.csv")
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
    
    return data, channels, time
