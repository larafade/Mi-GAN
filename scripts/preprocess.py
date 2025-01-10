"""
Author: Lara Fadel
Date: 2024-12-19
McGill Composite Center
Department of Chemical Engineering and Material Science, University of Southern California
Email: larafade@usc.edu

This script processes large microscopy images into smaller crops and performs data augmentation with optional GPU acceleration and rotation.
"""

import os
import yaml
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from mi_gan.utils.print_utils import Colors, Emojis

Image.MAX_IMAGE_PIXELS = None



class Preprocessor:
    def __init__(self, config_path):
        """
        Initializes the Preprocessor with a configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        print(f"{Colors.HEADER}{Emojis.CONFIG} Loading configuration from: {config_path}{Colors.ENDC}")
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.crop_size = self.config['preprocessing']['crop_size']
        self.overlap = self.config['preprocessing']['overlap']
        self.output_dir = os.path.join(os.getcwd(), self.config['preprocessing']['output_dir'])
        self.input_dir = os.path.join(os.getcwd(), self.config['preprocessing']['input_dir'])
        self.input_degree = self.config['preprocessing']['input_degree']
        self.output_degree = self.config['preprocessing']['output_degree']
        self.samples = self.config['preprocessing']['samples']
        self.file_type = self.config['preprocessing']['file_type']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"{Colors.OKCYAN}{Emojis.CONFIG} Initialized preprocessor with the following settings:{Colors.ENDC}")
        print(f"  Input directory: {Colors.OKGREEN}{self.input_dir}{Colors.ENDC}")
        print(f"  Output directory: {Colors.OKGREEN}{self.output_dir}{Colors.ENDC}")
        print(f"  Input degree: {Colors.OKBLUE}{self.input_degree}{Colors.ENDC}")
        print(f"  Output degree: {Colors.OKBLUE}{self.output_degree}{Colors.ENDC}")
        print(f"  File type: {Colors.OKCYAN}{self.file_type}{Colors.ENDC}")
        print(f"  Device: {Colors.OKGREEN}{self.device}{Colors.ENDC}")

        # Define transformations for augmentation
        augmentation_transforms = []
        if self.config['augmentation']['rotate']:
            augmentation_transforms.append(
                transforms.RandomRotation(self.config['augmentation']['rotation_range'])
            )
            print(f"{Colors.OKBLUE}{Emojis.PROCESSING} Rotation augmentation enabled with range: {self.config['augmentation']['rotation_range']}{Colors.ENDC}")
        if self.config['augmentation']['add_noise']:
            augmentation_transforms.append(transforms.Lambda(self.add_noise))
            print(f"{Colors.OKBLUE}{Emojis.PROCESSING} Noise augmentation enabled with noise level: {self.config['augmentation']['noise_level']}{Colors.ENDC}")

        self.transform = transforms.Compose(augmentation_transforms) if augmentation_transforms else None

    def add_noise(self, image_tensor):
        """
        Adds random noise to an image tensor.

        Args:
            image_tensor (torch.Tensor): Image tensor.

        Returns:
            torch.Tensor: Image tensor with noise added.
        """
        noise_level = self.config['augmentation'].get('noise_level', 0.1)
        noise = torch.randn_like(image_tensor) * noise_level
        return torch.clamp(image_tensor + noise, 0, 1)

    def preprocess_image(self, image_tensor):
        """
        Splits an image tensor into smaller crops.

        Args:
            image_tensor (torch.Tensor): Image tensor.

        Returns:
            List[torch.Tensor]: List of cropped image tensors.
        """
        crops = []
        c, h, w = image_tensor.shape
        step_size = int(self.crop_size * (1 - self.overlap))

        for top in range(0, h - self.crop_size + 1, step_size):
            for left in range(0, w - self.crop_size + 1, step_size):
                crop = image_tensor[:, top:top + self.crop_size, left:left + self.crop_size]
                crops.append(crop)
        return crops

    def run(self):
        """
        Executes the preprocessing pipeline for the specified input and output polishing degrees.
        """
        degrees = [self.input_degree, self.output_degree]

        for degree in degrees:
            degree_dir = os.path.join(self.input_dir, degree)
            print(f"{Colors.OKCYAN}{Emojis.LOADING} Loading images from: {degree_dir}{Colors.ENDC}")

            if not os.path.exists(degree_dir):
                print(f"{Colors.WARNING}{Emojis.ERROR} Directory not found: {degree_dir}. Skipping.{Colors.ENDC}")
                continue

            file_list = [
                file_name for file_name in os.listdir(degree_dir)
                if file_name.endswith(f'.{self.file_type}') and self._is_valid_sample(file_name, degree)
            ]

            if not file_list:
                print(f"{Colors.WARNING}{Emojis.SKIP} No valid files found for degree {degree} in {degree_dir}.{Colors.ENDC}")
                continue

            print(f"{Colors.OKGREEN}{Emojis.SUCCESS} Found {len(file_list)} files to process.{Colors.ENDC}")

            for file_name in tqdm(file_list, desc=f"Processing {degree} images"):
                image_path = os.path.join(degree_dir, file_name)
                print(f"{Colors.OKCYAN}  {Emojis.PROCESSING} Processing file: {file_name}{Colors.ENDC}")
                image = Image.open(image_path).convert("RGB")
                image_tensor = transforms.ToTensor()(image).to(self.device)

                save_dir = os.path.join(self.output_dir, degree)
                os.makedirs(save_dir, exist_ok=True)

                crops = self.preprocess_image(image_tensor)
                for i, crop in enumerate(crops):
                    augmented_crop = self.transform(crop) if self.transform else crop
                    output_path = os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}_{i}.{self.file_type}")
                    transforms.ToPILImage()(augmented_crop.cpu()).save(output_path)
                    
                print(f"{Colors.OKGREEN}    {Emojis.SAVE} Saved processed image: {file_name}{Colors.ENDC}")

    def _is_valid_sample(self, file_name, degree):
        """
        Checks if the file name matches the required samples and polishing degree.

        Args:
            file_name (str): The name of the file.
            degree (str): The polishing degree to match.

        Returns:
            bool: True if the file matches the criteria, False otherwise.
        """
        try:
            sample, file_degree = file_name.split('_')
            file_degree = file_degree.split('.')[0]  # Remove the extension
            is_valid_sample = (self.samples == "all" or sample in self.samples) and file_degree == degree
            if not is_valid_sample:
                print(f"{Colors.WARNING}      {Emojis.SKIP} Skipping file {file_name}: sample={sample}, degree={file_degree}, expected degree={degree}{Colors.ENDC}")
            return is_valid_sample
        except ValueError:
            print(f"{Colors.FAIL}      {Emojis.ERROR} Skipping file {file_name}: invalid format (expected format 'sample_degree.{self.file_type}').{Colors.ENDC}")
            return False


if __name__ == "__main__":
    preprocessor = Preprocessor('config/preprocess_config.yaml')
    preprocessor.run()
