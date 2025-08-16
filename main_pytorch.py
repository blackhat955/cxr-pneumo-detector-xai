#!/usr/bin/env python3
"""
PyTorch-based main script for chest X-ray pneumonia detection.
Compatible with Python 3.13 - no mutex lock issues.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from prepare_data import main as prepare_data_main
    from model_pytorch import ModelManager
    from train_pytorch import train_pneumonia_model
    from gradcam_pytorch import GradCAMVisualizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the project directory and have installed PyTorch dependencies")
    sys.exit(1)

class PneumoniaDetectionPipeline:
    """
    Complete pipeline for pneumonia detection using PyTorch.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_manager = None
        self.results = {}
        
        # Create directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('experiments', exist_ok=True)
        
        print("Pneumonia Detection Pipeline (PyTorch)")
        print("=" * 50)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration: {config['name']}")
    
    def prepare_data(self) -> bool:
        """
        Prepare and load the dataset.
        
        Returns:
            True if successful, False otherwise
        """
        print("\nStep 1: Data Preparation")
        print("-" * 30)
        
        try:
            import pickle
            
            # Check if processed data exists, if not prepare it
            processed_dir = self.config['processed_dir']
            data_files = [
                'X_train.npy', 'X_val.npy', 'X_test.npy',
                'y_train.npy', 'y_val.npy', 'y_test.npy'
            ]
            
            if not all(os.path.exists(os.path.join(processed_dir, f)) for f in data_files):
                print("📥 Preparing data...")
                prepare_data_main()
            
            # Load prepared data
            print("Loading processed data...")
            
            # Load processed data
            self.X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
            self.X_val = np.load(os.path.join(processed_dir, 'X_val.npy'))
            self.X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
            self.y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
            self.y_val = np.load(os.path.join(processed_dir, 'y_val.npy'))
            self.y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
            
            print(f"Data loaded successfully:")
            print(f"   Training: {len(self.X_train)} samples")
            print(f"   Validation: {len(self.X_val)} samples")
            print(f"   Test: {len(self.X_test)} samples")
            
            # Store data info
            self.results['data_info'] = {
                'train_samples': len(self.X_train),
                'val_samples': len(self.X_val),
                'test_samples': len(self.X_test),
                'image_shape': self.X_train.shape[1:],
                'num_classes': len(np.unique(self.y_train))
            }
            
            return True
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return False
    
    def train_model(self) -> bool:
        """
        Train the pneumonia detection model.
        
        Returns:
            True if successful, False otherwise
        """
        print("\nStep 2: Model Training")
        print("-" * 30)
        
        try:
            # Training configuration
            train_config = {
                'input_shape': self.config['input_shape'],
                'num_classes': 2,
                'batch_size': self.config['batch_size'],
                'epochs': self.config['epochs'],
                'learning_rate': self.config['learning_rate'],
                'weight_decay': self.config['weight_decay'],
                'dropout_rate': self.config['dropout_rate'],
                'patience': self.config['patience'],
                'save_path': os.path.join('experiments', f"{self.config['name']}_model.pth")
            }
            
            # Train model
            training_results = train_pneumonia_model(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                train_config
            )
            
            self.results['training'] = training_results
            self.model_path = training_results['model_path']
            
            print(f"Training completed successfully")
            print(f"Best validation accuracy: {training_results['best_accuracy']:.2f}%")
            print(f"Training time: {training_results['training_time']/60:.2f} minutes")
            
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def evaluate_model(self) -> bool:
        """
        Evaluate the trained model on test data.
        
        Returns:
            True if successful, False otherwise
        """
        print("\nStep 3: Model Evaluation")
        print("-" * 30)
        
        try:
            # Create model manager and load trained model
            self.model_manager = ModelManager(
                input_shape=self.config['input_shape'],
                num_classes=2
            )
            
            model = self.model_manager.load_model(self.model_path)
            
            # Evaluate on test data
            test_predictions = []
            test_probabilities = []
            
            print("Evaluating on test data...")
            for i in range(len(self.X_test)):
                pred, prob = self.model_manager.predict(model, self.X_test[i])
                test_predictions.append(pred[0])
                test_probabilities.append(prob[0])
            
            test_predictions = np.array(test_predictions)
            test_probabilities = np.array(test_probabilities)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
            
            test_accuracy = accuracy_score(self.y_test, test_predictions) * 100
            test_report = classification_report(self.y_test, test_predictions, 
                                              target_names=['Normal', 'Pneumonia'],
                                              output_dict=True)
            test_cm = confusion_matrix(self.y_test, test_predictions)
            test_auc = roc_auc_score(self.y_test, test_probabilities[:, 1])
            
            self.results['evaluation'] = {
                'test_accuracy': test_accuracy,
                'test_auc': test_auc,
                'classification_report': test_report,
                'confusion_matrix': test_cm.tolist()
            }
            
            print(f"Test accuracy: {test_accuracy:.2f}%")
            print(f"Test AUC: {test_auc:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, test_predictions, target_names=['Normal', 'Pneumonia']))
            
            return True
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return False
    
    def generate_explanations(self) -> bool:
        """
        Generate Grad-CAM explanations for sample images.
        
        Returns:
            True if successful, False otherwise
        """
        print("\nStep 4: Generating Explanations")
        print("-" * 30)
        
        try:
            if self.model_manager is None:
                self.model_manager = ModelManager(
                    input_shape=self.config['input_shape'],
                    num_classes=2
                )
            
            model = self.model_manager.load_model(self.model_path)
            visualizer = GradCAMVisualizer(self.model_manager)
            
            # Select sample images (2 from each class)
            normal_indices = np.where(self.y_test == 0)[0][:2]
            pneumonia_indices = np.where(self.y_test == 1)[0][:2]
            sample_indices = np.concatenate([normal_indices, pneumonia_indices])
            
            sample_images = [self.X_test[i] for i in sample_indices]
            sample_labels = [self.y_test[i] for i in sample_indices]
            
            # Generate explanations
            explanation_results = visualizer.batch_visualize(
                sample_images, model, 
                save_dir=os.path.join('experiments', f"{self.config['name']}_gradcam")
            )
            
            self.results['explanations'] = {
                'num_samples': len(sample_images),
                'sample_indices': sample_indices.tolist(),
                'sample_labels': sample_labels,
                'results': explanation_results
            }
            
            print(f"Generated explanations for {len(sample_images)} sample images")
            
            return True
            
        except Exception as e:
            print(f"Explanation generation error: {e}")
            return False
    
    def save_results(self) -> bool:
        """
        Save pipeline results.
        
        Returns:
            True if successful, False otherwise
        """
        print("\nStep 5: Saving Results")
        print("-" * 30)
        
        try:
            # Add pipeline metadata
            self.results['metadata'] = {
                'pipeline_name': 'Pneumonia Detection (PyTorch)',
                'config_name': self.config['name'],
                'timestamp': datetime.now().isoformat(),
                'python_version': sys.version,
                'pytorch_available': True,
                'tensorflow_available': False
            }
            
            # Save results
            results_path = os.path.join('experiments', f"{self.config['name']}_results.json")
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_for_json(self.results)
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"Results saved to {results_path}")
            
            # Print summary
            self._print_summary()
            
            return True
            
        except Exception as e:
            print(f"Results saving error: {e}")
            return False
    
    def _convert_for_json(self, obj):
        """
        Convert numpy arrays and other non-serializable objects for JSON.
        """
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj
    
    def _print_summary(self):
        """
        Print pipeline summary.
        """
        print("\nPipeline Summary")
        print("=" * 50)
        
        if 'data_info' in self.results:
            data_info = self.results['data_info']
            print(f"Dataset: {data_info['train_samples']} train, {data_info['val_samples']} val, {data_info['test_samples']} test")
        
        if 'training' in self.results:
            training = self.results['training']
            print(f"Training: {training['best_accuracy']:.2f}% best accuracy in {training['training_time']/60:.1f} min")
        
        if 'evaluation' in self.results:
            evaluation = self.results['evaluation']
            print(f"Test Results: {evaluation['test_accuracy']:.2f}% accuracy, {evaluation['test_auc']:.4f} AUC")
        
        if 'explanations' in self.results:
            explanations = self.results['explanations']
            print(f"Explanations: Generated for {explanations['num_samples']} samples")
        
        print(f"Model saved: {self.model_path}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_pipeline(self) -> bool:
        """
        Run the complete pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        steps = [
            ('Data Preparation', self.prepare_data),
            ('Model Training', self.train_model),
            ('Model Evaluation', self.evaluate_model),
            ('Generate Explanations', self.generate_explanations),
            ('Save Results', self.save_results)
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"\nPipeline failed at: {step_name}")
                return False
        
        total_time = time.time() - start_time
        print(f"\nPipeline completed successfully in {total_time/60:.2f} minutes")
        
        return True

def get_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Get predefined configurations.
    
    Returns:
        Dictionary of configurations
    """
    return {
        'quick': {
            'name': 'quick_test',
            'data_dir': 'data/raw',
            'processed_dir': 'data/processed',
            'input_shape': (32, 32),
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.01,
            'weight_decay': 1e-4,
            'dropout_rate': 0.3,
            'patience': 5
        },
        'production': {
            'name': 'production_model',
            'data_dir': 'data/raw',
            'processed_dir': 'data/processed',
            'input_shape': (224, 224),
            'batch_size': 64,
            'epochs': 100,
            'learning_rate': 0.0005,
            'weight_decay': 1e-4,
            'dropout_rate': 0.5,
            'patience': 15
        },
        'custom': {
            'name': 'custom_model',
            'data_dir': 'data/raw',
            'processed_dir': 'data/processed',
            'input_shape': (224, 224),
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'dropout_rate': 0.5,
            'patience': 10
        }
    }

def check_requirements() -> bool:
    """
    Check if required packages are available.
    
    Returns:
        True if all requirements are met
    """
    print("Checking requirements...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('medmnist', 'MedMNIST')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_pytorch.txt")
        return False
    
    print("All requirements satisfied")
    return True

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Pneumonia Detection Pipeline (PyTorch)')
    parser.add_argument('--config', choices=['quick', 'production', 'custom'], 
                       default='quick', help='Configuration to use')
    parser.add_argument('--check-only', action='store_true', 
                       help='Only check requirements')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (use existing model)')
    parser.add_argument('--model-path', type=str,
                       help='Path to existing model (if skipping training)')
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    if args.check_only:
        print("✅ Requirements check completed")
        return
    
    # Get configuration
    configs = get_configurations()
    config = configs[args.config]
    
    # Create and run pipeline
    pipeline = PneumoniaDetectionPipeline(config)
    
    if args.skip_training and args.model_path:
        pipeline.model_path = args.model_path
        # Skip to evaluation
        if pipeline.prepare_data():
            pipeline.evaluate_model()
            pipeline.generate_explanations()
            pipeline.save_results()
    else:
        success = pipeline.run_pipeline()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()