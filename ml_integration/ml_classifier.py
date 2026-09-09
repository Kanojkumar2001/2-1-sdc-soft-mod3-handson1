from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

class MLClassifier:
    def __init__(self, classifier_type='svm', **kwargs):
        self.classifier_type = classifier_type
        self.model = self._get_classifier(**kwargs)
        self.is_fitted = False
    
    def _get_classifier(self, **kwargs):
        """Get the appropriate classifier"""
        if self.classifier_type == 'svm':
            return SVC(**kwargs) if kwargs else SVC(kernel='rbf', C=1.0, gamma='scale')
        elif self.classifier_type == 'random_forest':
            return RandomForestClassifier(**kwargs) if kwargs else RandomForestClassifier(n_estimators=100)
        elif self.classifier_type == 'logistic_regression':
            return LogisticRegression(**kwargs) if kwargs else LogisticRegression(max_iter=1000)
        elif self.classifier_type == 'knn':
            return KNeighborsClassifier(**kwargs) if kwargs else KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"Unsupported classifier: {self.classifier_type}")
    
    def train(self, X_train, y_train):
        """Train the classifier"""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
    
    def predict(self, X_test):
        """Predict using the classifier"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the classifier"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet!")
        
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
    
    def save_model(self, path):
        """Save the trained model"""
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        """Load a trained model"""
        self.model = joblib.load(path)
        self.is_fitted = True
    
    def plot_confusion_matrix(self, conf_matrix, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.classifier_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{self.classifier_type}.png')
        plt.show()