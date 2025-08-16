import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import argparse
from datetime import datetime
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from model import create_model, compile_model, unfreeze_model, get_callbacks, print_model_summary
from prepare_data import create_data_generators

def load_processed_data(data_dir='data/processed'):
    """
    Load preprocessed data from disk
    """
    print("Loading processed data...")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    with open(os.path.join(data_dir, 'class_weights.pkl'), 'rb') as f:
        class_weights = pickle.load(f)
    
    print(f"Data loaded successfully!")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights

def plot_training_history(history, model_name, save_dir='experiments'):
    """
    Plot and save training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training & validation accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training & validation loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot training & validation AUC
    axes[1, 0].plot(history.history['auc'], label='Training AUC')
    axes[1, 0].plot(history.history['val_auc'], label='Validation AUC')
    axes[1, 0].set_title('Model AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot learning rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {save_dir}/{model_name}_training_history.png")

def evaluate_model(model, X_test, y_test, model_name, save_dir='experiments'):
    """
    Evaluate model on test set and generate comprehensive metrics
    """
    print("\nEvaluating model on test set...")
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_test_flat = y_test.flatten()
    y_pred_proba_flat = y_pred_proba.flatten()
    
    # Calculate metrics
    test_loss, test_accuracy, test_auc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    # Calculate additional metrics
    roc_auc = roc_auc_score(y_test_flat, y_pred_proba_flat)
    
    # Classification report
    class_report = classification_report(y_test_flat, y_pred, 
                                       target_names=['Normal', 'Pneumonia'], 
                                       output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_flat, y_pred)
    
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # Recall for positive class
    specificity = tn / (tn + fp)  # Recall for negative class
    
    # F1 scores
    f1_normal = class_report['Normal']['f1-score']
    f1_pneumonia = class_report['Pneumonia']['f1-score']
    f1_macro = class_report['macro avg']['f1-score']
    f1_weighted = class_report['weighted avg']['f1-score']
    
    # Print results
    print(f"\nTest Results for {model_name}:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"F1-Score (Pneumonia): {f1_pneumonia:.4f}")
    print(f"F1-Score (Normal): {f1_normal:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to JSON
    metrics = {
        'model_name': model_name,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'test_auc': float(test_auc),
        'roc_auc': float(roc_auc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1_pneumonia': float(f1_pneumonia),
        'f1_normal': float(f1_normal),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report
    }
    
    with open(os.path.join(save_dir, f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {save_dir}/{model_name}_metrics.json")
    print(f"Confusion matrix saved to {save_dir}/{model_name}_confusion_matrix.png")
    
    return metrics

def train_model(model_type='mobilenet', epochs_frozen=10, epochs_finetune=15, 
                batch_size=32, learning_rate=1e-3, data_dir='data/processed'):
    """
    Complete training pipeline with frozen and fine-tuning phases
    """
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{model_type}_{timestamp}"
    experiment_dir = f"experiments/{model_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Starting training for {model_name}")
    print(f"Experiment directory: {experiment_dir}")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights = load_processed_data(data_dir)
    
    # Create data generators
    train_generator, val_generator = create_data_generators(X_train, y_train, X_val, y_val, batch_size)
    
    # Create model
    print(f"\nCreating {model_type} model...")
    model, base_model = create_model(model_type)
    print_model_summary(model)
    
    # Phase 1: Train with frozen base model
    print("\n" + "="*50)
    print("PHASE 1: Training with frozen base model")
    print("="*50)
    
    model = compile_model(model, learning_rate=learning_rate, fine_tuning=False)
    callbacks = get_callbacks(f"{model_name}_frozen")
    
    history_frozen = model.fit(
        train_generator,
        epochs=epochs_frozen,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save frozen model
    model.save(os.path.join(experiment_dir, f'{model_name}_frozen.h5'))
    
    # Phase 2: Fine-tuning (if base_model exists)
    if base_model is not None and epochs_finetune > 0:
        print("\n" + "="*50)
        print("PHASE 2: Fine-tuning with unfrozen layers")
        print("="*50)
        
        # Unfreeze model
        model = unfreeze_model(model, base_model, trainable_layers=50)
        
        # Recompile with lower learning rate
        model = compile_model(model, learning_rate=learning_rate, fine_tuning=True)
        callbacks = get_callbacks(f"{model_name}_finetuned")
        
        # Continue training
        history_finetune = model.fit(
            train_generator,
            epochs=epochs_finetune,
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        combined_history = {}
        for key in history_frozen.history.keys():
            combined_history[key] = history_frozen.history[key] + history_finetune.history[key]
        
        # Create a mock history object
        class MockHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        combined_history_obj = MockHistory(combined_history)
    else:
        combined_history_obj = history_frozen
    
    # Save final model
    model.save(os.path.join(experiment_dir, f'{model_name}_final.h5'))
    
    # Plot training history
    plot_training_history(combined_history_obj, model_name, experiment_dir)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, model_name, experiment_dir)
    
    # Save training configuration
    config = {
        'model_type': model_type,
        'epochs_frozen': epochs_frozen,
        'epochs_finetune': epochs_finetune,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'total_epochs': epochs_frozen + epochs_finetune,
        'timestamp': timestamp,
        'data_dir': data_dir
    }
    
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining completed! Results saved to {experiment_dir}")
    
    return model, metrics, experiment_dir

def main():
    parser = argparse.ArgumentParser(description='Train chest X-ray classification model')
    parser.add_argument('--model_type', type=str, default='mobilenet', 
                       choices=['mobilenet', 'resnet', 'efficientnet_b0', 'efficientnet_b3', 'simple_cnn'],
                       help='Type of model to train')
    parser.add_argument('--epochs_frozen', type=int, default=10, 
                       help='Number of epochs to train with frozen base model')
    parser.add_argument('--epochs_finetune', type=int, default=15, 
                       help='Number of epochs for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    
    args = parser.parse_args()
    
    # Enable mixed precision if GPU is available
    if tf.config.list_physical_devices('GPU'):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled")
    
    # Train model
    model, metrics, experiment_dir = train_model(
        model_type=args.model_type,
        epochs_frozen=args.epochs_frozen,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir
    )
    
    print("\nTraining completed successfully!")
    print(f"Best AUC: {metrics['test_auc']:.4f}")
    print(f"Best Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")

if __name__ == "__main__":
    main()