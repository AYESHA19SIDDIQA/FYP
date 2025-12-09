import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class EEGDataset(Dataset):
    def __init__(self, root_dir, indexes=None, target_length=None):
        """
        Args:
            root_dir (string): Directory with all the EEG data and data.csv.
            indexes (list of int, optional): Subset of annotation indices to use.
            target_length (int, optional): Number of time steps to crop each sample to.
        """
        annotations_file = os.path.join(root_dir, 'data.csv')
        self.annotations = pd.read_csv(annotations_file)
        if indexes is not None:
            self.annotations = self.annotations[self.annotations.index.isin(indexes)].reset_index(drop=True)
        self.root_dir = root_dir
        self.target_length = target_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load EEG array
        eeg_path = self.annotations.iloc[idx, 0]
        eeg = np.load(eeg_path, allow_pickle=True)['data']  # shape: (C, T)
        # Crop to target length if specified
        if self.target_length is not None:
            eeg = eeg[:, :self.target_length]
        # Convert to tensor
        eeg = torch.tensor(eeg).float()

        # Load and one-hot encode label
        label_raw = self.annotations.iloc[idx, 1]
        if label_raw == 0:
            label = torch.tensor([1, 0], dtype=torch.float)
        else:
            label = torch.tensor([0, 1], dtype=torch.float)

        return eeg, label


class KFoldDataset():
    def __init__(self, root_dir, n_splits=10, shuffle=True, random_state=42):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the data.
        """
        annotations_file = os.path.join(root_dir, 'data.csv')
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotations_file)

        # Filter to include only rows where "train" is mentioned in the "File" column
        self.annotations = self.annotations[self.annotations["File"].str.contains("train", case=False, na=False)]
        
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.gen = self.skf.split(self.annotations["File"], self.annotations["Label"])

    def __iter__(self):
        return self

    def __next__(self):
        curr = next(self.gen)
        return EEGDataset(self.root_dir, curr[0]), EEGDataset(self.root_dir, curr[1])
