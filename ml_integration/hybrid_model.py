import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .feature_extractor import FeatureExtractor
from .ml_classifier import MLClassifier
import os

class HybridModel:
    """Hybrid model combining CNN feature extraction with ML classification"""
    
    def __init__(self, config):
        self.config = config
        self.feature_extractor = FeatureExtractor(
            model_type=config['ml']['feature_extraction'],
            use_pytorch=False
        )
        self.ml_classifier = MLClassifier(
            classifier_type=config['ml']['classifier']
        )
        self.scaler = StandardScaler()
        self.pca = None
        self.class_names = []
    
    def extract_features_from_dataset(self, data_generator):
        """Extract CNN features from dataset"""
        features = []
        labels = []
        
        for _ in range(len(data_generator)):
            batch_images, batch_labels = next(data_generator)
            batch_features = []
            for image in batch_images:
                # Convert to 0-255 range if normalized
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                feature = self.feature_extractor.extract_features(image)
                batch_features.append(feature)
            features.extend(batch_features)
            labels.extend(np.argmax(batch_labels, axis=1))
        
        return np.array(features), np.array(labels)
    
    def train(self, train_generator, val_generator=None):
        """Train the hybrid model"""
        print("Extracting features from training data...")
        X_train, y_train = self.extract_features_from_dataset(train_generator)
        self.class_names = list(train_generator.class_indices.keys())
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Optional PCA for dimensionality reduction
        if X_train_scaled.shape[1] > 1000:
            self.pca = PCA(n_components=0.95)  # Keep 95% variance
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
        
        print("Training ML classifier...")
        self.ml_classifier.train(X_train_scaled, y_train)
        
        # Validation
        if val_generator is not None and len(val_generator) > 0:
            print("Evaluating on validation data...")
            X_val, y_val = self.extract_features_from_dataset(val_generator)
            X_val_scaled = self.scaler.transform(X_val)
            if self.pca:
                X_val_scaled = self.pca.transform(X_val_scaled)
            results = self.ml_classifier.evaluate(X_val_scaled, y_val)
            print(f"Validation Accuracy: {results['accuracy']:.4f}")
            return results

        return self.ml_classifier.evaluate(X_train_scaled, y_train)
    
    def predict(self, image_path):
        """Predict class for a single image"""
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        features = self.feature_extractor.extract_features(image)
        features_scaled = self.scaler.transform([features])
        if self.pca:
            features_scaled = self.pca.transform(features_scaled)
        
        prediction = self.ml_classifier.predict(features_scaled)
        return self.class_names[prediction[0]]
    
    def save_model(self, path):
        """Save the hybrid model"""
        os.makedirs(path, exist_ok=True)
        self.ml_classifier.save_model(os.path.join(path, 'ml_classifier.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        if self.pca:
            joblib.dump(self.pca, os.path.join(path, 'pca.pkl'))
        joblib.dump(self.class_names, os.path.join(path, 'class_names.pkl'))
    
    def load_model(self, path):
        """Load the hybrid model"""
        self.ml_classifier.load_model(os.path.join(path, 'ml_classifier.pkl'))
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        if os.path.exists(os.path.join(path, 'pca.pkl')):
            self.pca = joblib.load(os.path.join(path, 'pca.pkl'))
        self.class_names = joblib.load(os.path.join(path, 'class_names.pkl'))