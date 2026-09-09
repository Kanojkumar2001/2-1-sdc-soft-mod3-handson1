import os
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
            train_datagen, self.config['data']['train_path'], shuffle=True
        )
        val_generator = self._create_generator(
            val_datagen, self.config['data']['val_path'], shuffle=False
        )
        
        return train_generator, val_generator

    def _create_generator(self, datagen, directory, shuffle):
        """Load either class subdirectories or filenames prefixed by a class label."""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        if not any(
            entry.is_file() and entry.name.lower().endswith(image_extensions)
            for entry in os.scandir(directory)
        ) and not any(
            any(
                child.is_file() and child.name.lower().endswith(image_extensions)
                for child in os.scandir(class_dir.path)
            )
            for class_dir in os.scandir(directory)
            if class_dir.is_dir()
        ):
            return None

        entries = [
            entry for entry in os.scandir(directory)
            if entry.is_file() and entry.name.lower().endswith(
                image_extensions
            )
        ]

        if entries:
            records = []
            for entry in entries:
                if '_' not in entry.name:
                    continue
                label = entry.name.split('_', 1)[0]
                records.append({'filename': entry.path, 'class': label})

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
        class_names = os.listdir(self.config['data']['train_path'])
        
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(self.config['data']['train_path'], class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append(img_path)
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