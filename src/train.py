import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from model import CircadianModel
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_curve, auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# restructuring table from 56x14 -> 112x9
def restructure_feat_table(table):
    new_rows = []

    for _, row in table.iterrows():

        subject = row.iloc[0]
        eye_cond = row.iloc[1]

        eye_encoded = 0 if eye_cond == "EC" else 1

        normal_features = row.iloc[2:8].values
        deprived_features = row.iloc[8:14].values

        # Normal sleep
        new_rows.append(
            [subject, 0, eye_encoded] + list(normal_features)
        )

        # Sleep deprived
        new_rows.append(
            [subject, 1, eye_encoded] + list(deprived_features)
        )

    new_table = pd.DataFrame(
        new_rows,
        columns=[
            "Subject",
            "SleepCondition",
            "EyeState",
            "Alpha",
            "Theta",
            "ThetaAlphaRatio",
            "PAF",
            "Slope",
            "Entropy"
        ]
    )
    return new_table

class SpectralDataset(Dataset):
    def __init__(self, dataframe, feature_cols, include_eye=True):
        
        self.subjects = dataframe["Subject"].values
        self.labels = torch.tensor(
            dataframe["SleepCondition"].values,
            dtype=torch.float32
        )
        
        if include_eye:
            self.features = torch.tensor(
            dataframe[feature_cols + ["EyeState"]].values,
            dtype=torch.float32
            )
        else:
            self.features = torch.tensor(
                dataframe[feature_cols].values,
                dtype=torch.float32
            )
        
    def __len__(self):
        return len(self.labels)
    
    def __get_item__(self, idx):
        return self.features[idx], self.labels[idx]


# restructuring the data
features = restructure_feat_table(pd.read_csv('data/eeg_psd_features.csv'))

# standardising the features
feature_cols = [
    "Alpha",
    "Theta",
    "ThetaAlphaRatio",
    "PAF",
    "Slope",
    "Entropy"
]

scaler = StandardScaler()
features[feature_cols] = scaler.fit_transform(features[feature_cols])

dataset = SpectralDataset(features, feature_cols)

X = dataset.features.numpy()
y = dataset.labels.numpy()
groups = features["Subject"].values

logo = LeaveOneGroupOut()


# Instantiation and training set up

torch.manual_seed(67)

StasisNet = CircadianModel(input_dim=7).to(device)
optimiser = torch.optim.Adam(StasisNet.parameters(), lr=0.01)
loss_func = nn.BCEWithLogitsLoss()

all_fold_accuracies = []
train_loss = []
val_loss = []

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    
    X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).to(device)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32).to(device)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32).to(device)

    
    # Training
    n_epochs = 1000
    for epoch in range(n_epochs):
        StasisNet.train()
        optimiser.zero_grad()
        outputs = StasisNet(X_train).squeeze()
        loss = loss_func(outputs, y_train)
        loss.backward()
        optimiser.step()
        
        train_loss.append(loss.item())
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    # Evaluation
    StasisNet.eval()
    with torch.no_grad():
        logits = StasisNet(X_test).squeeze()
        probs = torch.sigmoid(logits)
        
        y_true = y_test.cpu()

        preds = (probs > 0.5).float()
        accuracy = (preds == y_test).float().mean().item()
        all_fold_accuracies.append(accuracy)
        print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")
print(f"\nAverage Accuracy across subjects: {np.mean(all_fold_accuracies):.4f}")


