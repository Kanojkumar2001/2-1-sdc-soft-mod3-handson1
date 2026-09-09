import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from src.data_loader import DataLoader
from src.model import create_model
from src.trainer import CNNTrainer

def main():
    os.chdir(PROJECT_ROOT)

    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    data_loader = DataLoader(config)
    train_generator, val_generator = data_loader.load_data_tensorflow()
    
    # Create model
    model = create_model(
        model_type=config['model']['architecture'],
        num_classes=config['data']['num_classes']
    )
    
    # Train model
    trainer = CNNTrainer(model, config)
    history = trainer.train_tensorflow(train_generator, val_generator)
    
    # Plot training history
    trainer.plot_training_history()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()