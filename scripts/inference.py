"""
Author: Lara Fadel
Date: 2024-12-20
MCGill Composite Center
Department of Chemical Engineering and Material Science, University of Southern California
Email: larafade@usc.edu

This script performs full-image inference using a trained GAN model.
"""

import os
from PIL import Image
from torchvision import transforms
from mi_gan.utils.config_utils import load_config, ConfigObject
from mi_gan.models.test_model import TestModel


class Inferencer:
    def __init__(self, config_path):
        """
        Initializes the Inferencer with a configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        print("⚙️  Loading configuration...")
        config_dict = load_config(config_path)
        self.config = ConfigObject(config_dict)  # Convert config to ConfigObject

        self.model = TestModel(self.config)

        # Determine model weights to load
        self.run_name = self.config.inference.run_name
        self.load_dir = self.config.inference.load_dir

        if self.load_dir:
            # Load weights from the specified directory
            self.load_dir = os.path.abspath(self.load_dir)
            print(f"📂 Loading pretrained model from: {self.load_dir}")
            self.model.load_networks_from_dir(self.load_dir)
        elif self.run_name:
            # Load the latest checkpoint from a specified run
            print(f"📂 Loading latest checkpoint from logs/{self.run_name}")
            checkpoints_dir = os.path.abspath(os.path.join('logs', self.run_name, 'checkpoints'))
            self.load_latest_checkpoint(checkpoints_dir)
        else:
            # Auto-detect the latest run and checkpoint
            print("🔄 Auto-detecting the latest run and checkpoint...")
            latest_run_dir = self.find_latest_run_dir()
            if not latest_run_dir:
                raise ValueError("❌ No valid runs found in logs.")
            checkpoints_dir = os.path.join(latest_run_dir, 'checkpoints')
            print(f"📂 Found latest run: {latest_run_dir}")
            self.load_latest_checkpoint(checkpoints_dir)

        # Resolve input and output directories relative to the working directory
        self.input_dir = os.path.abspath(self.config.inference.input_dir)
        self.output_dir = os.path.abspath(self.config.inference.output_dir)

    def find_latest_run_dir(self):
        """
        Finds the latest run directory in the logs folder.

        Returns:
            str: Path to the latest run directory.
        """
        logs_dir = os.path.abspath('logs')
        if not os.path.exists(logs_dir):
            return None

        # Get all directories in the logs folder
        run_dirs = [os.path.join(logs_dir, d) for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
        if not run_dirs:
            return None

        # Sort directories by creation/modification time
        latest_run = max(run_dirs, key=os.path.getmtime)
        return latest_run

    def load_latest_checkpoint(self, checkpoints_dir):
        """
        Finds and loads the latest checkpoint in the specified directory.

        Args:
            checkpoints_dir (str): Path to the checkpoints directory.
        """
        if not os.path.exists(checkpoints_dir):
            raise ValueError(f"❌ Checkpoints directory does not exist: {checkpoints_dir}")

        # Find the latest checkpoint file
        checkpoint_files = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
        if not checkpoint_files:
            raise ValueError(f"❌ No checkpoint files found in {checkpoints_dir}.")

        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"🔄 Loading checkpoint: {latest_checkpoint}")
        self.model.load_networks_from_dir(checkpoints_dir, specific_file=latest_checkpoint)

    def run_inference(self):
        """
        Executes inference on all images specified in the configuration file.
        """
        print(f"🔍 Running inference on images in: {self.input_dir}")
        print(f"📂 Saving results to: {self.output_dir}")

        os.makedirs(self.output_dir, exist_ok=True)
        # Define transformations (same as used during training)
        preprocess = transforms.Compose([
            transforms.Resize((self.config.inference.crop_size, self.config.inference.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Assuming normalization was used during training
        ])

        # Resolve all image paths relative to the working directory
        images = [img for img in os.listdir(self.input_dir) if img.endswith('.png') or img.endswith('.jpg')]

        if not images:
            raise ValueError(f"❌ No valid image files found in {self.input_dir}.")

        for image_name in images:
            print(f"🖼️  Processing image: {image_name}")
            input_path = os.path.join(self.input_dir, image_name)
            output_path = os.path.join(self.output_dir, f"inferred_{image_name}")

            # Open and preprocess input image
            input_image = Image.open(input_path).convert('RGB')
            input_tensor = preprocess(input_image).unsqueeze(0).to(self.model.device)

            # Set the model's input and run the forward pass
            self.model.set_input({'A': input_tensor, 'A_paths': input_path})  # Include A_paths to avoid KeyError
            self.model.test()
            output_tensor = self.model.fake.cpu().squeeze(0)  # Get output tensor and remove batch dimension

            # Postprocess and save the output image
            output_image = transforms.ToPILImage()(output_tensor)
            output_image.save(output_path)
            print(f"✅ Saved inferred image: {output_path}")


if __name__ == "__main__":
    print("🚀 Initializing Inferencer...")
    inferencer = Inferencer('config/inference_config.yaml')
    inferencer.run_inference()
