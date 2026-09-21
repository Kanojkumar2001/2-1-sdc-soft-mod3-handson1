import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import joblib
from src.data_loader import DataLoader
from ml_integration.hybrid_model import HybridModel

def main():
    os.chdir(PROJECT_ROOT)
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    data_loader = DataLoader(config)
    train_generator, val_generator = data_loader.load_data_tensorflow()
    
    # Initialize and train hybrid model
    hybrid_model = HybridModel(config)
    results = hybrid_model.train(train_generator, val_generator)
    
    # Save model
    hybrid_model.save_model('models/hybrid_model')
    
    print("Hybrid model training completed successfully!")
    print(f"Validation Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])

if __name__ == "__main__":
    main()