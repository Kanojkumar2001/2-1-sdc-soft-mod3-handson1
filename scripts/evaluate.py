import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import tensorflow as tf
import torch
from src.data_loader import DataLoader
from src.model import create_model, PyTorchCNN
from src.evaluator import ModelEvaluator
from ml_integration.hybrid_model import HybridModel
from src.utils import Utils

def evaluate_cnn_tensorflow(config_path='config/config.yaml'):
    """Evaluate TensorFlow CNN model"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    data_loader = DataLoader(config)
    _, _, test_generator = data_loader.load_data_tensorflow()
    
    # Load model
    model = create_model(
        model_type=config['model']['architecture'],
        num_classes=config['data']['num_classes']
    )
    
    # Load best weights
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.h5')
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print("Warning: No saved weights found. Using untrained model.")
    
    # Evaluate
    evaluator = ModelEvaluator(model, config)
    results = evaluator.evaluate_tensorflow(test_generator)
    
    # Generate plots and reports
    evaluator.plot_confusion_matrix('evaluation_results/tensorflow_confusion_matrix.png')
    evaluator.plot_per_class_accuracy('evaluation_results/tensorflow_per_class_accuracy.png')
    evaluator.generate_evaluation_report('evaluation_results/tensorflow_report.txt')
    
    return results

def evaluate_cnn_pytorch(config_path='config/config.yaml'):
    """Evaluate PyTorch CNN model"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    data_loader = DataLoader(config)
    _, _, test_loader = data_loader.load_data_pytorch()
    
    # Load model
    model = PyTorchCNN(num_classes=config['data']['num_classes'])
    
    # Load best weights
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        device = Utils.get_device()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print("Warning: No saved weights found. Using untrained model.")
    
    # Evaluate
    evaluator = ModelEvaluator(model, config)
    results = evaluator.evaluate_pytorch(test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate plots and reports
    evaluator.plot_confusion_matrix('evaluation_results/pytorch_confusion_matrix.png')
    evaluator.plot_per_class_accuracy('evaluation_results/pytorch_per_class_accuracy.png')
    evaluator.generate_evaluation_report('evaluation_results/pytorch_report.txt')
    
    return results

def evaluate_hybrid(config_path='config/config.yaml'):
    """Evaluate hybrid model"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    data_loader = DataLoader(config)
    _, _, test_generator = data_loader.load_data_tensorflow()
    
    # Load hybrid model
    hybrid_model = HybridModel(config)
    model_path = 'models/hybrid_model'
    
    if os.path.exists(model_path):
        hybrid_model.load_model(model_path)
        print(f"Loaded hybrid model from {model_path}")
    else:
        print("Warning: No saved hybrid model found. Training new model...")
        hybrid_model.train(test_generator)
    
    # Evaluate
    evaluator = ModelEvaluator(hybrid_model, config)
    results = evaluator.evaluate_hybrid(test_generator, hybrid_model)
    
    # Generate plots and reports
    evaluator.plot_confusion_matrix('evaluation_results/hybrid_confusion_matrix.png')
    evaluator.plot_per_class_accuracy('evaluation_results/hybrid_per_class_accuracy.png')
    evaluator.generate_evaluation_report('evaluation_results/hybrid_report.txt')
    
    return results

def evaluate_all_models(config_path='config/config.yaml'):
    """Evaluate all models and compare results"""
    print("=" * 60)
    print("EVALUATING ALL MODELS")
    print("=" * 60)
    
    # Create evaluation results directory
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Evaluate each model
    results = {}
    
    print("\n1. Evaluating TensorFlow CNN...")
    print("-" * 40)
    results['TensorFlow CNN'] = evaluate_cnn_tensorflow(config_path)
    
    print("\n2. Evaluating PyTorch CNN...")
    print("-" * 40)
    results['PyTorch CNN'] = evaluate_cnn_pytorch(config_path)
    
    print("\n3. Evaluating Hybrid Model...")
    print("-" * 40)
    results['Hybrid Model'] = evaluate_hybrid(config_path)
    
    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    # Create comparison table
    print(f"\n{'Model':<20} {'Test Accuracy':<15} {'Test Loss':<15}")
    print("-" * 50)
    for name, res in results.items():
        accuracy = res.get('test_accuracy', 'N/A')
        loss = res.get('test_loss', 'N/A')
        print(f"{name:<20} {accuracy:<15} {loss:<15}")
    
    # Generate comparison plot
    evaluator = ModelEvaluator(None)
    evaluator.compare_models(results, 'evaluation_results/model_comparison.png')
    
    print("\nAll evaluation results saved to 'evaluation_results' directory")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Image Classification Models')
    parser.add_argument('--model', type=str, choices=['tensorflow', 'pytorch', 'hybrid', 'all'],
                       default='all', help='Model type to evaluate')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    if args.model == 'tensorflow':
        evaluate_cnn_tensorflow(args.config)
    elif args.model == 'pytorch':
        evaluate_cnn_pytorch(args.config)
    elif args.model == 'hybrid':
        evaluate_hybrid(args.config)
    else:
        evaluate_all_models(args.config)

if __name__ == "__main__":
    main()