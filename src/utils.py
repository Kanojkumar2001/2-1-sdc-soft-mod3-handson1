import os
import random
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import yaml

class Utils:
    """Utility functions for the image classification project"""
    
    @staticmethod
    def set_seed(seed=42):
        """Set random seed for reproducibility"""
        import torch

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    @staticmethod
    def get_device():
        """Get available device"""
        import torch

        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    @staticmethod
    def count_parameters(model):
        """Count trainable parameters in a model"""
        if isinstance(model, tf.keras.Model):
            return model.count_params()
        try:
            import torch
            is_pytorch_model = isinstance(model, torch.nn.Module)
        except (ImportError, OSError):
            is_pytorch_model = False
        if is_pytorch_model:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return 0
    
    @staticmethod
    def preprocess_image(image_path, target_size=(224, 224), normalize=True):
        """
        Preprocess a single image for inference
        
        Args:
            image_path: Path to image
            target_size: Target size (height, width)
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image array
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, target_size)
        
        # Normalize
        if normalize:
            image = image / 255.0
        
        return image
    
    @staticmethod
    def display_predictions(model, image_paths, class_names, model_type='tensorflow'):
        """
        Display predictions for multiple images
        
        Args:
            model: Trained model
            image_paths: List of image paths
            class_names: List of class names
            model_type: 'tensorflow' or 'pytorch'
        """
        n_images = len(image_paths)
        fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))
        if n_images == 1:
            axes = [axes]
        
        for idx, image_path in enumerate(image_paths):
            # Load and preprocess image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get prediction
            processed_image = Utils.preprocess_image(image_path)
            
            if model_type == 'tensorflow':
                # Add batch dimension
                input_tensor = np.expand_dims(processed_image, axis=0)
                predictions = model.predict(input_tensor, verbose=0)[0]
                pred_class = np.argmax(predictions)
                confidence = np.max(predictions)
            else:  # PyTorch
                import torch
                from torchvision import transforms
                
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                pil_image = Image.open(image_path).convert('RGB')
                input_tensor = transform(pil_image).unsqueeze(0)
                
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predictions = probabilities.cpu().numpy()[0]
                    pred_class = np.argmax(predictions)
                    confidence = np.max(predictions)
            
            # Display image and prediction
            axes[idx].imshow(image_rgb)
            axes[idx].axis('off')
            title = f"{class_names[pred_class]}\n({confidence:.2f})"
            axes[idx].set_title(title, fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def save_config(config, save_path='config_saved.yaml'):
        """Save configuration to YAML file"""
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Configuration saved to {save_path}")
    
    @staticmethod
    def load_config(config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def get_timestamp():
        """Get current timestamp as string"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def create_experiment_dir(base_dir='experiments'):
        """Create experiment directory with timestamp"""
        timestamp = Utils.get_timestamp()
        exp_dir = os.path.join(base_dir, f'exp_{timestamp}')
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir
    
    @staticmethod
    def log_metrics(metrics, log_file):
        """Log metrics to file"""
        with open(log_file, 'a') as f:
            timestamp = Utils.get_timestamp()
            f.write(f"\n[{timestamp}]\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
    
    @staticmethod
    def visualize_feature_maps(model, image_path, layer_name=None, save_path='feature_maps.png'):
        """
        Visualize feature maps from a CNN layer
        
        Args:
            model: TensorFlow model
            image_path: Path to input image
            layer_name: Name of the layer to visualize (if None, last conv layer)
            save_path: Path to save visualization
        """
        # Preprocess image
        image = Utils.preprocess_image(image_path, normalize=True)
        image = np.expand_dims(image, axis=0)
        
        # Get intermediate layer outputs
        if layer_name is None:
            # Find the last convolutional layer
            for layer in reversed(model.layers):
                if 'conv' in layer.name:
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            raise ValueError("No convolutional layer found")
        
        # Create intermediate model
        intermediate_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        
        # Get feature maps
        feature_maps = intermediate_model.predict(image, verbose=0)[0]
        
        # Visualize feature maps
        n_filters = feature_maps.shape[-1]
        n_cols = 8
        n_rows = int(np.ceil(n_filters / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        
        for i in range(n_filters):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.imshow(feature_maps[:, :, i], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Filter {i+1}', fontsize=8)
        
        # Hide unused subplots
        for i in range(n_filters, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].axis('off')
            else:
                axes[col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"Feature maps saved to {save_path}")