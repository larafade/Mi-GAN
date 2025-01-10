"""
Author: Lara Fadel
Date: 2024-12-19
MCGill Composite Center
Department of Chemical Engineering and Material Science, University of Southern California
Email: larafade@usc.edu

Provides functions for data loading and augmentation.
"""

import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class MicroscopyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset with image paths and optional transformations.

        Args:
            data_dir (str): Path to the directory containing images.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def create_dataloader(config, mode='train'):
    """
    Creates a DataLoader for training, validation, or testing.

    Args:
        config (dict): Configuration parameters.
        mode (str): Mode of the DataLoader ('train', 'val', 'test').

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    data_dir = config[f'{mode}_dir']
    batch_size = config['batch_size']
    shuffle = (mode == 'train')

    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = MicroscopyDataset(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config['num_workers'])
