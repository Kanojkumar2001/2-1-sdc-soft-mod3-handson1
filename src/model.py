import tensorflow as tf
from tensorflow.keras import layers, models

class CNNModel(tf.keras.Model):
    """CNN Model for Image Classification"""
    def __init__(self, num_classes=3):
        super(CNNModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

class PyTorchCNN:
    """PyTorch CNN Model"""
    def __init__(self, num_classes=3):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        self._torch = torch
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        import torch.nn.functional as F
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ResNetBlock(tf.keras.Model):
    """ResNet Block for deeper networks"""
    def __init__(self, filters):
        super(ResNetBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, (3, 3), padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, (3, 3), padding='same')
        self.bn2 = layers.BatchNormalization()
        self.shortcut = layers.Conv2D(filters, (1, 1), padding='same')
        
    def call(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x + shortcut)
        return x

def create_model(model_type='cnn', num_classes=3):
    """Factory function to create models"""
    if model_type.lower() == 'cnn':
        return CNNModel(num_classes)
    elif model_type.lower() == 'resnet':
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        for _ in range(2):
            x = ResNetBlock(64)(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(num_classes, activation='softmax')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)