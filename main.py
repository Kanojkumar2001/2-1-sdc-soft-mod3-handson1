import sys
import os
import yaml
import argparse

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data/train', 'data/validation', 'data/test',
        'models/saved_models', 'models/model_checkpoints',
        'notebooks'
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Image Classification Project')
    parser.add_argument('--mode', choices=['train_cnn', 'train_hybrid', 'predict'],
                       default='train_cnn', help='Mode to run')
    parser.add_argument('--image', type=str, help='Image path for prediction')
    parser.add_argument('--model', type=str, choices=['cnn', 'hybrid'], 
                       default='hybrid', help='Model type for prediction')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    if args.mode == 'train_cnn':
        from scripts.train_cnn import main as train_cnn
        train_cnn()
    elif args.mode == 'train_hybrid':
        from scripts.train_hybrid import main as train_hybrid
        train_hybrid()
    elif args.mode == 'predict':
        if not args.image:
            print("Please provide an image path for prediction")
            return
        from scripts.predict import main as predict
        sys.argv = [sys.argv[0], args.image, args.model]
        predict()

if __name__ == "__main__":
    main()