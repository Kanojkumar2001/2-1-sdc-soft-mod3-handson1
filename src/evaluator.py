import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import os

class ModelEvaluator:
    """Comprehensive model evaluation class for both TensorFlow and PyTorch models"""
    
    def __init__(self, model, config=None):
        """
        Initialize the evaluator
        
        Args:
            model: Trained model (TensorFlow or PyTorch)
            config: Configuration dictionary
        """
        self.model = model
        self.config = config or {}
        self.results = {}
        self.class_names = []
        
    def evaluate_tensorflow(self, test_generator, save_results=True):
        """
        Evaluate TensorFlow model on test data
        
        Args:
            test_generator: TensorFlow data generator for test set
            save_results: Boolean to save results to file
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating TensorFlow model on test data...")
        
        # Get class names
        self.class_names = list(test_generator.class_indices.keys())
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Calculate metrics
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': predicted_classes,
            'true_labels': true_classes,
            'prediction_probabilities': predictions
        }
        
        # Generate classification report
        results['classification_report'] = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        results['confusion_matrix'] = confusion_matrix(true_classes, predicted_classes)
        
        # Store results
        self.results = results
        
        # Print results
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        # Save results if requested
        if save_results:
            self.save_results()
        
        return results
    
    def evaluate_pytorch(self, test_loader, device='cuda', save_results=True):
        """
        Evaluate PyTorch model on test data
        
        Args:
            test_loader: PyTorch DataLoader for test set
            device: Device to run evaluation on
            save_results: Boolean to save results to file
            
        Returns:
            Dictionary with evaluation metrics
        """
        import torch

        print("Evaluating PyTorch model on test data...")
        
        self.model.eval()
        self.model.to(device)
        
        test_loss = 0
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate accuracy
        test_accuracy = np.mean(all_predictions == all_true_labels)
        avg_test_loss = test_loss / len(test_loader)
        
        # Get class names if available
        if hasattr(test_loader.dataset, 'class_names'):
            self.class_names = test_loader.dataset.class_names
        
        # Calculate metrics
        results = {
            'test_loss': avg_test_loss,
            'test_accuracy': test_accuracy,
            'predictions': all_predictions,
            'true_labels': all_true_labels,
            'prediction_probabilities': all_probabilities
        }
        
        # Generate classification report
        results['classification_report'] = classification_report(
            all_true_labels, 
            all_predictions, 
            target_names=self.class_names if self.class_names else None,
            output_dict=True
        )
        
        # Generate confusion matrix
        results['confusion_matrix'] = confusion_matrix(all_true_labels, all_predictions)
        
        # Store results
        self.results = results
        
        # Print results
        print(f"\nTest Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_true_labels, all_predictions, 
                                  target_names=self.class_names if self.class_names else None))
        
        # Save results if requested
        if save_results:
            self.save_results()
        
        return results
    
    def evaluate_hybrid(self, test_generator, hybrid_model, save_results=True):
        """
        Evaluate hybrid model (CNN + ML classifier) on test data
        
        Args:
            test_generator: Data generator for test set
            hybrid_model: Trained HybridModel instance
            save_results: Boolean to save results to file
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating Hybrid model on test data...")
        
        # Extract features from test data
        from ml_integration.hybrid_model import HybridModel
        if isinstance(hybrid_model, HybridModel):
            X_test, y_test = hybrid_model.extract_features_from_dataset(test_generator)
            # Scale features
            X_test_scaled = hybrid_model.scaler.transform(X_test)
            if hybrid_model.pca:
                X_test_scaled = hybrid_model.pca.transform(X_test_scaled)
            
            # Get predictions from ML classifier
            predictions = hybrid_model.ml_classifier.predict(X_test_scaled)
            self.class_names = hybrid_model.class_names
            
        else:
            raise ValueError("hybrid_model must be an instance of HybridModel")
        
        # Calculate metrics
        test_accuracy = np.mean(predictions == y_test)
        
        results = {
            'test_accuracy': test_accuracy,
            'predictions': predictions,
            'true_labels': y_test,
        }
        
        # Generate classification report
        results['classification_report'] = classification_report(
            y_test, 
            predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, predictions)
        
        # Store results
        self.results = results
        
        # Print results
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=self.class_names))
        
        # Save results if requested
        if save_results:
            self.save_results()
        
        return results
    
    def plot_confusion_matrix(self, save_path='confusion_matrix.png', normalize=False):
        """
        Plot confusion matrix
        
        Args:
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
        """
        if 'confusion_matrix' not in self.results:
            raise ValueError("No confusion matrix found. Run evaluate() first.")
        
        conf_matrix = self.results['confusion_matrix']
        
        if normalize:
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_roc_curves(self, save_path='roc_curves.png'):
        """
        Plot ROC curves for multi-class classification
        
        Args:
            save_path: Path to save the plot
        """
        if 'prediction_probabilities' not in self.results:
            raise ValueError("No prediction probabilities found. Run evaluate() first.")
        
        y_true = self.results['true_labels']
        y_prob = self.results['prediction_probabilities']
        
        # Convert to one-hot encoding if needed
        if len(y_true.shape) == 1:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        else:
            y_true_bin = y_true
        
        plt.figure(figsize=(10, 8))
        
        for i in range(len(self.class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"ROC curves saved to {save_path}")
    
    def plot_per_class_accuracy(self, save_path='per_class_accuracy.png'):
        """
        Plot per-class accuracy
        
        Args:
            save_path: Path to save the plot
        """
        if 'classification_report' not in self.results:
            raise ValueError("No classification report found. Run evaluate() first.")
        
        report = self.results['classification_report']
        
        # Extract per-class metrics
        class_names = []
        accuracies = []
        
        for class_name in self.class_names:
            if class_name in report:
                class_names.append(class_name)
                accuracies.append(report[class_name]['precision'])
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_names, accuracies, color='skyblue')
        plt.xlabel('Classes')
        plt.ylabel('Precision Score')
        plt.title('Per-Class Precision')
        plt.ylim([0, 1])
        
        # Add value labels on bars
        for bar, value in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"Per-class accuracy plot saved to {save_path}")
    
    def generate_evaluation_report(self, save_path='evaluation_report.txt'):
        """
        Generate comprehensive evaluation report
        
        Args:
            save_path: Path to save the report
        """
        if not self.results:
            raise ValueError("No results found. Run evaluate() first.")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nTest Accuracy: {self.results.get('test_accuracy', 'N/A')}")
        report_lines.append(f"Test Loss: {self.results.get('test_loss', 'N/A')}")
        
        if 'classification_report' in self.results:
            report_lines.append("\n" + "=" * 40)
            report_lines.append("CLASSIFICATION REPORT")
            report_lines.append("=" * 40)
            
            report = self.results['classification_report']
            
            # Per-class metrics
            report_lines.append("\nPer-Class Metrics:")
            report_lines.append("-" * 60)
            report_lines.append(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
            report_lines.append("-" * 60)
            
            for class_name in self.class_names:
                if class_name in report:
                    metrics = report[class_name]
                    report_lines.append(
                        f"{class_name:<20} {metrics['precision']:.3f}     {metrics['recall']:.3f}     "
                        f"{metrics['f1-score']:.3f}     {metrics['support']:<10}"
                    )
            
            # Overall metrics
            if 'accuracy' in report:
                report_lines.append("-" * 60)
                report_lines.append(f"\nOverall Accuracy: {report['accuracy']:.3f}")
                if 'macro avg' in report:
                    report_lines.append(f"Macro Average F1-Score: {report['macro avg']['f1-score']:.3f}")
                if 'weighted avg' in report:
                    report_lines.append(f"Weighted Average F1-Score: {report['weighted avg']['f1-score']:.3f}")
        
        # Confusion matrix
        if 'confusion_matrix' in self.results:
            report_lines.append("\n" + "=" * 40)
            report_lines.append("CONFUSION MATRIX")
            report_lines.append("=" * 40)
            report_lines.append("\n" + str(self.results['confusion_matrix']))
        
        # Write to file
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Evaluation report saved to {save_path}")
        
        # Also print to console
        print('\n'.join(report_lines))
    
    def save_results(self, directory='evaluation_results'):
        """
        Save all evaluation results to files
        
        Args:
            directory: Directory to save results
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save classification report as CSV
        if 'classification_report' in self.results:
            report = self.results['classification_report']
            df = pd.DataFrame(report).transpose()
            df.to_csv(os.path.join(directory, 'classification_report.csv'))
        
        # Save confusion matrix as CSV
        if 'confusion_matrix' in self.results:
            np.savetxt(os.path.join(directory, 'confusion_matrix.csv'), 
                      self.results['confusion_matrix'], delimiter=',')
        
        # Generate plots
        try:
            self.plot_confusion_matrix(os.path.join(directory, 'confusion_matrix.png'))
            self.plot_roc_curves(os.path.join(directory, 'roc_curves.png'))
            self.plot_per_class_accuracy(os.path.join(directory, 'per_class_accuracy.png'))
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
        
        # Generate report
        self.generate_evaluation_report(os.path.join(directory, 'evaluation_report.txt'))
        
        print(f"\nAll results saved to '{directory}' directory")
    
    def compare_models(self, model_results_dict, save_path='model_comparison.png'):
        """
        Compare multiple models
        
        Args:
            model_results_dict: Dictionary with model names as keys and results dictionaries as values
            save_path: Path to save the comparison plot
        """
        plt.figure(figsize=(12, 6))
        
        model_names = list(model_results_dict.keys())
        accuracies = [model_results_dict[name]['test_accuracy'] for name in model_names]
        
        bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.xlabel('Models')
        plt.ylabel('Test Accuracy')
        plt.title('Model Performance Comparison')
        plt.ylim([0, 1])
        
        # Add value labels on bars
        for bar, value in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"Model comparison saved to {save_path}")