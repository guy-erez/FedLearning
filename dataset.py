import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class DemoDataset(Dataset):

    def __init__(self, csv_file, label='death', drop = ['reference_date_internal', 'site'], transform=None):
        self.label = label
        self.x = pd.read_csv(csv_file)
        self.x.drop(columns = drop, inplace = True)
        self.x['gender'] = LabelEncoder().fit_transform(self.x['gender'])  # TODO: make it general for all string entries 
        self.y = self.x.pop(label)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.x.iloc[idx]
        x = np.array([x], dtype=float)
        y = self.y.iloc[idx]
        y = np.array([y], dtype=float)

        sample = {'x': x, 'y': y}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        x, y = sample['x'], sample['y']

        return {'x': torch.from_numpy(x),
                'y': torch.from_numpy(y)}