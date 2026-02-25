import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, X, y, regime_labels=None):
        """
        X : numpy array or tensor (N, features)
        y : numpy array or tensor (N,)
        regime_labels : optional (N,)
        """

        # Convert to FloatTensor
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

        if regime_labels is not None:
            self.regime_labels = torch.FloatTensor(regime_labels)
        else:
            self.regime_labels = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.regime_labels is not None:
            return self.X[idx], self.y[idx], self.regime_labels[idx]
        else:
            return self.X[idx], self.y[idx]
        

def walk_forward_split(X, y, regime_labels=None):
        n = len(X)

        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        X_train = X[:train_end]
        y_train = y[:train_end]

        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]

        X_test = X[val_end:]
        y_test = y[val_end:]

        if regime_labels is not None:
            r_train = regime_labels[:train_end]
            r_val = regime_labels[train_end:val_end]
            r_test = regime_labels[val_end:]
        else:
            r_train = r_val = r_test = None

        return (X_train, y_train, r_train,
                X_val, y_val, r_val,
                X_test, y_test, r_test)
    
    
def create_dataloaders(X, y, regime_labels=None, batch_size=32):

        (X_train, y_train, r_train,
        X_val, y_val, r_val,
        X_test, y_test, r_test) = walk_forward_split(X, y, regime_labels)

        train_dataset = FinancialTimeSeriesDataset(X_train, y_train, r_train)
        val_dataset = FinancialTimeSeriesDataset(X_val, y_val, r_val)
        test_dataset = FinancialTimeSeriesDataset(X_test, y_test, r_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print("Dataset Sizes:")
        print(f"Train: {len(train_dataset)}")
        print(f"Validation: {len(val_dataset)}")
        print(f"Test: {len(test_dataset)}")

        return train_loader, val_loader, test_loader

