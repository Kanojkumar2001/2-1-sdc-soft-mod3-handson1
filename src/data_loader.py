import os
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image

class ImageDataset:
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.image_size = config['data']['image_size']
        self.batch_size = config['data']['batch_size']
        self.project_root = Path(__file__).resolve().parent.parent

    def _resolve_data_path(self, configured_path):
        """Resolve config paths independently of the current working directory."""
        path = Path(configured_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    @staticmethod
    def _has_images(directory):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
        return any(
            entry.is_file() and entry.suffix.lower() in image_extensions
            for entry in directory.iterdir()
        ) or any(
            child.is_file() and child.suffix.lower() in image_extensions
            for class_dir in directory.iterdir()
            if class_dir.is_dir()
            for child in class_dir.iterdir()
        )
        
    def load_data_tensorflow(self):
        """Load data using TensorFlow's ImageDataGenerator"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = self._create_generator(
            train_datagen, self._resolve_data_path(self.config['data']['train_path']), shuffle=True
        )
        val_generator = self._create_generator(
            val_datagen, self._resolve_data_path(self.config['data']['val_path']), shuffle=False,
            required=False
        )
        
        return train_generator, val_generator

    def _create_generator(self, datagen, directory, shuffle, required=True):
        """Load either class subdirectories or filenames prefixed by a class label."""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        directory = Path(directory)
        if not directory.exists():
            if required:
                raise FileNotFoundError(
                    f"Dataset directory not found: {directory}. "
                    "Check the paths in config/config.yaml."
                )
            return None
        if not directory.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {directory}")
        if not self._has_images(directory):
            if required:
                raise ValueError(f"No image files found in dataset directory: {directory}")
            return None

        entries = [
            entry for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() in image_extensions
        ]

        if entries:
            records = []
            for entry in entries:
                if '_' not in entry.name:
                    continue
                label = entry.name.split('_', 1)[0]
                records.append({'filename': str(entry), 'class': label})

            if not records:
                raise ValueError(
                    f"No labelled images found in '{directory}'. "
                    "Flat datasets must use '<class>_<image>.<extension>' filenames."
                )

            dataframe = pd.DataFrame(records)
            return datagen.flow_from_dataframe(
                dataframe,
                x_col='filename',
                y_col='class',
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=shuffle,
            )

        return datagen.flow_from_directory(
            directory,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=shuffle,
        )
    
    def load_data_pytorch(self):
        """Load data using PyTorch DataLoader"""
        import torch
        from torch.utils.data import Dataset, DataLoader as TorchDataLoader
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load images and labels
        images = []
        labels = []
        train_path = self._resolve_data_path(self.config['data']['train_path'])
        if not train_path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {train_path}. "
                "Check the paths in config/config.yaml."
            )

        class_names = sorted(
            path.name for path in train_path.iterdir()
            if path.is_dir() and self._has_images(path)
        )
        if not class_names:
            raise ValueError(
                f"No class folders containing images found in: {train_path}"
            )
        
        for class_idx, class_name in enumerate(class_names):
            class_path = train_path / class_name
            for image_path in class_path.iterdir():
                if image_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'):
                    continue
                images.append(str(image_path))
                labels.append(class_idx)
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )
        
        class TorchImageDataset(Dataset):
            def __init__(self, image_paths, labels, transform=None):
                self.image_paths = image_paths
                self.labels = labels
                self.transform = transform

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                image = Image.open(self.image_paths[idx]).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, self.labels[idx]

        train_dataset = TorchImageDataset(train_paths, train_labels, transform=transform)
        val_dataset = TorchImageDataset(val_paths, val_labels, transform=transform)
        
        train_loader = TorchDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader