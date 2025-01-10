import os
import random
import shutil
from PIL import Image
from tqdm import tqdm
import yaml


class DatasetCreator:
    def __init__(self, config_path):
        """
        Initializes the DatasetCreator with a configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        print(f"🔄 Loading configuration from: {config_path}")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        preprocessing_config = config['preprocessing']
        self.preprocessed_dir = os.path.join(os.getcwd(), preprocessing_config['output_dir'])
        self.input_degree = preprocessing_config['input_degree']
        self.output_degree = preprocessing_config['output_degree']
        self.save_dir = os.path.join(os.getcwd(), "dataset")
        self.train_ratio = preprocessing_config['train_ratio']
        self.val_ratio = preprocessing_config['val_ratio']
        self.test_ratio = preprocessing_config['test_ratio']
        self.file_type = preprocessing_config['file_type']

        # Ensure ratios sum to 1
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1, "Train, val, and test ratios must sum to 1."

    def create_aligned_dataset(self):
        """
        Aligns preprocessed images horizontally for input and output polishing degrees.
        """
        input_path = os.path.join(self.preprocessed_dir, self.input_degree)
        output_path = os.path.join(self.preprocessed_dir, self.output_degree)
        aligned_path = os.path.join(self.save_dir, "aligned")

        os.makedirs(aligned_path, exist_ok=True)

        input_files = sorted(f for f in os.listdir(input_path) if f.endswith(f".{self.file_type}"))
        output_files = sorted(f for f in os.listdir(output_path) if f.endswith(f".{self.file_type}"))

        assert len(input_files) == len(output_files), "Input and output directories must contain the same number of files."

        print(f"🔄 Aligning {len(input_files)} preprocessed images...")
        for input_file, output_file in tqdm(zip(input_files, output_files), desc="Aligning images", total=len(input_files)):
            input_image = Image.open(os.path.join(input_path, input_file)).convert("RGB")
            output_image = Image.open(os.path.join(output_path, output_file)).convert("RGB")

            # Ensure images are of the same size
            assert input_image.size == output_image.size, f"Image sizes do not match: {input_file} and {output_file}"

            # Create a new image wide enough to stack horizontally
            aligned_image = Image.new("RGB", (input_image.width * 2, input_image.height))
            aligned_image.paste(input_image, (0, 0))
            aligned_image.paste(output_image, (input_image.width, 0))

            aligned_image.save(os.path.join(aligned_path, input_file))

    def split_dataset(self):
        """
        Splits the aligned dataset into train, validation, and test sets.
        """
        aligned_path = os.path.join(self.save_dir, "aligned")
        train_path = os.path.join(self.save_dir, "train")
        val_path = os.path.join(self.save_dir, "val")
        test_path = os.path.join(self.save_dir, "test")

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        files = sorted(os.listdir(aligned_path))
        random.shuffle(files)

        total_files = len(files)
        train_split = int(total_files * self.train_ratio)
        val_split = train_split + int(total_files * self.val_ratio)

        train_files = files[:train_split]
        val_files = files[train_split:val_split]
        test_files = files[val_split:]

        self._copy_files(train_files, aligned_path, train_path, "Training")
        self._copy_files(val_files, aligned_path, val_path, "Validation")
        self._copy_files(test_files, aligned_path, test_path, "Testing")

        # Clean up the aligned folder after splitting
        self._delete_folder(aligned_path)
        print(f"🗑️ Aligned folder deleted: {aligned_path}")

        # Clean up the processed folder
        self._delete_folder(self.preprocessed_dir)
        print(f"🗑️ Processed folder deleted: {self.preprocessed_dir}")

    def _copy_files(self, file_list, src_dir, dest_dir, split_name):
        """
        Copies files to the destination directory.

        Args:
            file_list (list): List of filenames to copy.
            src_dir (str): Source directory.
            dest_dir (str): Destination directory.
            split_name (str): Name of the dataset split (e.g., "Training").
        """
        print(f"📂 {split_name}: Copying {len(file_list)} files to {dest_dir}")
        for file in tqdm(file_list, desc=f"Copying {split_name} files"):
            shutil.copy2(os.path.join(src_dir, file), os.path.join(dest_dir, file))

    def _delete_folder(self, folder_path):
        """
        Deletes a folder and its contents.

        Args:
            folder_path (str): Path to the folder to delete.
        """
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)


if __name__ == "__main__":
    dataset_creator = DatasetCreator('config/preprocess_config.yaml')
    print("🔄 Aligning preprocessed dataset...")
    dataset_creator.create_aligned_dataset()
    print("✅ Alignment complete. Splitting dataset...")
    dataset_creator.split_dataset()
    print("🎉 Dataset creation complete!")
