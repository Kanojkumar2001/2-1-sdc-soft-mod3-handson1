import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import cv2
import numpy as np
import tensorflow as tf
from src.model import create_model
from ml_integration.hybrid_model import HybridModel

def predict_cnn(image_path, model_path):
    """Predict using CNN model"""
    # Load model
    model = create_model(num_classes=3)
    model.load_weights(model_path)
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Predict
    predictions = model.predict(image)
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return class_idx, confidence

def predict_hybrid(image_path, model_path):
    """Predict using hybrid model"""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    hybrid_model = HybridModel(config)
    hybrid_model.load_model(model_path)
    
    prediction = hybrid_model.predict(image_path)
    return prediction, 1.0

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'data/test/sample.jpg'
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'hybrid'
    
    if model_type == 'cnn':
        class_idx, confidence = predict_cnn(image_path, 'models/saved_models/best_model.h5')
        print(f"Predicted class: {class_idx}, Confidence: {confidence:.4f}")
    else:
        prediction, _ = predict_hybrid(image_path, 'models/hybrid_model')
        print(f"Predicted class: {prediction}")

if __name__ == "__main__":
    main()