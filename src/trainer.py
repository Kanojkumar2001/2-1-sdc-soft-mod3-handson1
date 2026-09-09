import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class CNNTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.history = {'train_loss': [], 'val_loss': [], 
                       'train_acc': [], 'val_acc': []}
    
    def train_tensorflow(self, train_generator, val_generator):
        """Train using TensorFlow"""
        optimizer_name = self.config['model']['optimizer']
        
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['model']['learning_rate'])
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.config['model']['learning_rate'])
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config['model']['learning_rate'])
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = []
        if self.config['training']['early_stopping'] and val_generator is not None:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    patience=self.config['training']['patience'],
                    restore_best_weights=True
                )
            )
        
        if self.config['training']['save_best'] and val_generator is not None:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.config['training']['checkpoint_dir'], 'best_model.h5'),
                    save_best_only=True
                )
            )
        
        fit_args = {
            'x': train_generator,
            'epochs': self.config['model']['epochs'],
            'callbacks': callbacks,
            'verbose': 1,
        }
        if val_generator is not None:
            fit_args['validation_data'] = val_generator

        history = self.model.fit(**fit_args)
        
        self.history['train_loss'] = history.history['loss']
        self.history['val_loss'] = history.history.get('val_loss', [])
        self.history['train_acc'] = history.history['accuracy']
        self.history['val_acc'] = history.history.get('val_accuracy', [])
        
        return history
    
    def train_pytorch(self, train_loader, val_loader, device='cuda'):
        """Train using PyTorch"""
        import torch
        import torch.optim as optim
        
        self.model = self.model.to(device)
        optimizer = getattr(optim, self.config['model']['optimizer'].upper())(
            self.model.parameters(), 
            lr=self.config['model']['learning_rate']
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        best_val_acc = 0
        
        for epoch in range(self.config['model']['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 
                          os.path.join(self.config['training']['checkpoint_dir'], 'best_model.pth'))
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()