"""
Author: Lara Fadel
Date: 2024-12-19
MCGill Composite Center
Department of Chemical Engineering and Material Science, University of Southern California
Email: larafade@usc.edu

This script trains a GAN model for microscopy image denoising using paired datasets.
"""

import os
import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from mi_gan.utils.config_utils import ConfigObject
from mi_gan.models import create_model
from torch.utils.tensorboard import SummaryWriter
import yaml


class PairedDataset(Dataset):
    """
    A simple PyTorch Dataset for paired input-output images.
    """
    def __init__(self, dataroot, phase, transform=None):
        """
        Args:
            dataroot (str): Path to the dataset directory.
            phase (str): Phase of the dataset (train, val, or test).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.dataroot = os.path.join(dataroot, phase)
        self.image_paths = sorted(os.listdir(self.dataroot))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataroot, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")

        # Split the image into input and output (assumes horizontal concatenation)
        width, height = image.size
        input_image = image.crop((0, 0, width // 2, height))
        target_image = image.crop((width // 2, 0, width, height))

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return {
            'A': input_image,  # Input image
            'B': target_image,  # Target image
            'A_paths': image_path,  # Input image path
            'B_paths': image_path   # Target image path (same as input for paired dataset)
        }


class Trainer:
    def __init__(self, config_path):
        """
        Initialize the trainer with the configuration file.
        """
        print(f"🔄 Loading configuration from: {config_path}")
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine run_name and log_dir
        base_run_name = self.config['model'].get('name', 'default_run')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{base_run_name}_{timestamp}"
        log_dir = os.path.join(self.config['training']['log_dir'], run_name)
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.config['training']['log_dir'] = log_dir
        self.config['training']['checkpoints_dir'] = checkpoint_dir

        # Initialize dataset, DataLoader, model, and logger
        transform = transforms.Compose([
            transforms.Resize((self.config['data']['crop_size'], self.config['data']['crop_size'])),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.train_loader = DataLoader(
            PairedDataset(self.config['data']['dataroot'], "train", transform=transform),
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_threads']
        )

        self.val_loader = DataLoader(
            PairedDataset(self.config['data']['dataroot'], "val", transform=transform),
            batch_size=1,
            shuffle=False,
            num_workers=self.config['data']['num_threads']
        )

        # Merge relevant sections into one object and initialize model
        merged_config = {**self.config['model'], **self.config['training']}
        self.model = create_model(ConfigObject(merged_config))
        self.model.setup(ConfigObject(merged_config))

        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"✅ Trainer initialized on device: {self.device} with logs saved to: {log_dir}")

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def train(self):
        """
        Train the model using the training dataset and validate periodically.
        """
        total_epochs = self.config['model']['n_epochs'] + self.config['model']['n_epochs_decay']
        for epoch in range(1, total_epochs + 1):
            print(f"🔄 Starting Epoch {epoch}/{total_epochs}")
            self.model.train()
            epoch_loss = 0

            # Use tqdm to wrap the training DataLoader
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch}/{total_epochs}", unit="batch") as pbar:
                for i, data in enumerate(self.train_loader):
                    self.model.set_input(data)
                    self.model.optimize_parameters()
                    losses = self.model.get_current_losses()

                    # Log each training loss separately
                    for loss_name, loss_value in losses.items():
                        self.writer.add_scalar(f"Train/Losses/{loss_name}", loss_value, global_step=epoch * len(self.train_loader) + i)

                    # Optionally accumulate generator-related losses
                    if 'G_GAN' in losses and 'G_L1' in losses and 'G_BCE' in losses:
                        epoch_loss += losses['G_GAN'] + losses['G_L1'] + losses['G_BCE']

                    # Update tqdm progress bar
                    pbar.set_postfix(losses)
                    pbar.update(1)

            avg_train_loss = epoch_loss / len(self.train_loader)
            print(f"✅ Epoch {epoch} Complete - Avg Train Loss: {avg_train_loss}")

            # Validation
            if epoch % self.config['training']['save_epoch_freq'] == 0 or epoch == total_epochs:
                val_loss = self.validate(epoch)
                print(f"🔍 Validation Loss after Epoch {epoch}: {val_loss}")
                self.writer.add_scalar("Validation/Avg_Validation_Loss", val_loss, global_step=epoch)

                # Save model
                self.model.save_networks(epoch)
                print(f"💾 Model checkpoints saved for Epoch {epoch}")
                

    def validate(self, epoch):
        """
        Validate the model using the validation dataset.

        Args:
            epoch (int): Current epoch for logging.

        Returns:
            float: Average validation loss.
        """
        print("🔍 Starting Validation...")
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for data in self.val_loader:
                self.model.set_input(data)
                self.model.test()

                # Calculate and log validation losses
                losses = self.model.get_current_losses()
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f"Validation/Losses/{loss_name}", loss_value, global_step=epoch)

                # Optionally accumulate generator-related losses
                if 'G_GAN' in losses and 'G_L1' in losses and 'G_BCE' in losses:
                    val_loss += losses['G_GAN'] + losses['G_L1'] + losses['G_BCE']

        avg_val_loss = val_loss / len(self.val_loader)
        print(f"✅ Validation Complete - Avg Validation Loss: {avg_val_loss}")
        return avg_val_loss


if __name__ == "__main__":
    trainer = Trainer(config_path="config/train_config.yaml")
    trainer.train()
