import torch
from torch.utils.data import Dataset

class GeneratedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        X = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1) / 255.0  # Convert to PyTorch format [C, H, W]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return X, y
