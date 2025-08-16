#!/usr/bin/env python3
"""
PyTorch-based training script for chest X-ray pneumonia detection.
Compatible with Python 3.13 - no mutex lock issues.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm
import os
import json
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

from model_pytorch import ModelManager, ChestXrayCNN

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class ModelTrainer:
    """
    PyTorch model trainer with comprehensive training pipeline.
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.device = model_manager.device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Average loss and accuracy for the epoch
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Validate for one epoch.
        
        Returns:
            Average loss, accuracy, predictions, and true labels
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', leave=False)
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, np.array(all_predictions), np.array(all_targets)
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   epochs: int = 50, learning_rate: float = 0.001, weight_decay: float = 1e-4,
                   patience: int = 10, save_path: str = 'experiments/pytorch_model.pth') -> Dict:
        """
        Train the model with comprehensive monitoring.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            save_path: Path to save the best model
            
        Returns:
            Training history and metrics
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=patience)
        
        # Training loop
        start_time = time.time()
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch(model, val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start
            print(f"\n📈 Results:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"🎯 New best validation accuracy: {best_val_acc:.2f}%")
                
                # Save model
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                metadata = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc,
                    'learning_rate': current_lr,
                    'timestamp': datetime.now().isoformat()
                }
                self.model_manager.save_model(model, save_path, metadata)
            
            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"\n🏁 Training completed in {total_time/60:.2f} minutes")
        print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
        
        # Final evaluation
        final_metrics = self.evaluate_model(model, val_loader, val_targets, val_preds)
        
        return {
            'history': self.history,
            'best_val_acc': best_val_acc,
            'total_time': total_time,
            'final_metrics': final_metrics
        }
    
    def evaluate_model(self, model: nn.Module, val_loader: DataLoader, 
                      true_labels: np.ndarray, predictions: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation.
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model performance...")
        
        # Get probabilities for ROC curve
        model.eval()
        all_probs = []
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                output = model(data)
                probs = torch.softmax(output, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = (predictions == true_labels).mean() * 100
        
        # Classification report
        class_report = classification_report(true_labels, predictions, 
                                           target_names=['Normal', 'Pneumonia'],
                                           output_dict=True)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(true_labels, all_probs)
            fpr, tpr, _ = roc_curve(true_labels, all_probs)
        except:
            roc_auc = 0.0
            fpr, tpr = [0, 1], [0, 1]
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        # Print results
        print(f"\n🎯 Final Results:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"\n📋 Classification Report:")
        print(classification_report(true_labels, predictions, target_names=['Normal', 'Pneumonia']))
        
        return metrics
    
    def plot_training_history(self, save_path: str = 'experiments/training_history.png'):
        """
        Plot training history.
        """
        if not self.history['train_loss']:
            print("⚠️ No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(epochs, self.history['learning_rate'], 'g-')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss difference plot
        loss_diff = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
        axes[1, 1].plot(epochs, loss_diff, 'purple')
        axes[1, 1].set_title('Overfitting Monitor (Val Loss - Train Loss)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled to prevent hanging in non-interactive environments
        plt.close()  # Close the figure to free memory
        
        print(f"📈 Training history saved to {save_path}")

def train_pneumonia_model(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         config: Dict = None) -> Dict:
    """
    Main training function for pneumonia detection model.
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        config: Training configuration
        
    Returns:
        Training results and model path
    """
    # Default configuration
    default_config = {
        'input_shape': (28, 28),
        'num_classes': 2,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'dropout_rate': 0.5,
        'patience': 10,
        'save_path': 'experiments/pytorch_pneumonia_model.pth'
    }
    
    if config:
        default_config.update(config)
    
    config = default_config
    
    print("Training Pneumonia Detection Model with PyTorch")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Configuration: {config}")
    
    # Create model manager and model
    model_manager = ModelManager(
        input_shape=config['input_shape'],
        num_classes=config['num_classes']
    )
    
    model = model_manager.create_model(dropout_rate=config['dropout_rate'])
    
    # Create data loaders
    train_loader, val_loader = model_manager.create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=config['batch_size']
    )
    
    # Create trainer and train model
    trainer = ModelTrainer(model_manager)
    
    results = trainer.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        patience=config['patience'],
        save_path=config['save_path']
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save training results
    results_path = config['save_path'].replace('.pth', '_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    print(f"Training results saved to {results_path}")
    
    return {
        'model_path': config['save_path'],
        'results_path': results_path,
        'best_accuracy': results['best_val_acc'],
        'training_time': results['total_time']
    }

if __name__ == "__main__":
    # Test training with dummy data
    print("Testing PyTorch training pipeline...")
    
    # Create dummy data
    X_train = np.random.rand(100, 28, 28, 1)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.rand(20, 28, 28, 1)
    y_val = np.random.randint(0, 2, 20)
    
    # Quick test configuration
    test_config = {
        'epochs': 2,
        'batch_size': 16,
        'save_path': 'experiments/test_pytorch_model.pth'
    }
    
    try:
        results = train_pneumonia_model(X_train, y_train, X_val, y_val, test_config)
        print("PyTorch training pipeline test passed!")
        print(f"Results: {results}")
    except Exception as e:
        print(f"PyTorch training pipeline test failed: {e}")