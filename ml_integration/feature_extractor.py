import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2

class FeatureExtractor:
    def __init__(self, model_type='vgg16', use_pytorch=False):
        model_type = 'vgg16' if model_type == 'cnn_features' else model_type
        self.model_type = model_type
        self.use_pytorch = use_pytorch
        
        if not use_pytorch:
            if model_type == 'vgg16':
                base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            elif model_type == 'resnet50':
                base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            
            self.model = tf.keras.Model(inputs=base_model.input, 
                                       outputs=base_model.output)
        else:
            import torch
            from torchvision import models, transforms
            self.torch = torch
            if model_type == 'vgg16':
                self.model = models.vgg16(pretrained=True)
            elif model_type == 'resnet50':
                self.model = models.resnet50(pretrained=True)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def extract_features(self, image):
        """Extract features from an image"""
        if not self.use_pytorch:
            # TensorFlow feature extraction
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            image = cv2.resize(image, (224, 224))
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            features = self.model.predict(image, verbose=0)
            return features.flatten()
        else:
            # PyTorch feature extraction
            image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            
            with self.torch.no_grad():
                features = self.model(image_tensor)
            return features.numpy().flatten()