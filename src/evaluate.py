import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, calibration_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import json
import pickle
from scipy import stats
import pandas as pd
from gradcam import GradCAM

class ModelEvaluator:
    def __init__(self, model_path, data_dir='data/processed'):
        """
        Initialize model evaluator
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.model = None
        self.gradcam = None
        
        # Load model
        self.load_model()
        
        # Load data
        self.load_data()
        
        # Initialize GradCAM
        self.gradcam = GradCAM(self.model)
    
    def load_model(self):
        """
        Load trained model
        """
        print(f"Loading model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
    
    def load_data(self):
        """
        Load test data
        """
        print("Loading test data...")
        self.X_test = np.load(os.path.join(self.data_dir, 'X_test.npy'))
        self.y_test = np.load(os.path.join(self.data_dir, 'y_test.npy')).flatten()
        
        # Also load validation data for calibration
        self.X_val = np.load(os.path.join(self.data_dir, 'X_val.npy'))
        self.y_val = np.load(os.path.join(self.data_dir, 'y_val.npy')).flatten()
        
        print(f"Test data shape: {self.X_test.shape}")
        print(f"Validation data shape: {self.X_val.shape}")
    
    def get_predictions(self, X):
        """
        Get model predictions
        """
        y_pred_proba = self.model.predict(X, verbose=0)
        y_pred_proba = y_pred_proba.flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return y_pred, y_pred_proba
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curve
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
        
        return fpr, tpr, thresholds, auc_score
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot Precision-Recall curve
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap_score = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {ap_score:.3f})')
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='red', linestyle='--', 
                   label=f'Random classifier (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        
        plt.show()
        
        return precision, recall, thresholds, ap_score
    
    def plot_calibration_curve(self, y_true, y_pred_proba, save_path=None, n_bins=10):
        """
        Plot calibration curve to assess probability calibration
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label="Model", color='blue')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot (Reliability Curve)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration curve saved to {save_path}")
        
        plt.show()
        
        return fraction_of_positives, mean_predicted_value
    
    def calibrate_probabilities(self, method='platt'):
        """
        Calibrate model probabilities using Platt scaling or Isotonic regression
        """
        print(f"Calibrating probabilities using {method} scaling...")
        
        # Get validation predictions for calibration
        _, y_val_proba = self.get_predictions(self.X_val)
        
        if method.lower() == 'platt':
            # Platt scaling (logistic regression)
            calibrator = LogisticRegression()
        elif method.lower() == 'isotonic':
            # Isotonic regression
            calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            raise ValueError("Method must be 'platt' or 'isotonic'")
        
        # Fit calibrator on validation set
        calibrator.fit(y_val_proba.reshape(-1, 1), self.y_val)
        
        # Get calibrated test predictions
        _, y_test_proba = self.get_predictions(self.X_test)
        y_test_proba_calibrated = calibrator.predict_proba(y_test_proba.reshape(-1, 1))[:, 1]
        
        return calibrator, y_test_proba_calibrated
    
    def find_optimal_threshold(self, y_true, y_pred_proba, metric='f1'):
        """
        Find optimal threshold based on specified metric
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                from sklearn.metrics import f1_score
                score = f1_score(y_true, y_pred_thresh)
            elif metric == 'sensitivity':
                # True Positive Rate
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
                score = tp / (tp + fn) if (tp + fn) > 0 else 0
            elif metric == 'specificity':
                # True Negative Rate
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
                score = tn / (tn + fp) if (tn + fp) > 0 else 0
            elif metric == 'youden':
                # Youden's J statistic (Sensitivity + Specificity - 1)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            else:
                raise ValueError("Metric must be 'f1', 'sensitivity', 'specificity', or 'youden'")
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        print(f"Optimal threshold for {metric}: {optimal_threshold:.3f} (score: {optimal_score:.3f})")
        
        return optimal_threshold, optimal_score, thresholds, scores
    
    def plot_threshold_analysis(self, y_true, y_pred_proba, save_path=None):
        """
        Plot threshold analysis for different metrics
        """
        metrics = ['f1', 'sensitivity', 'specificity', 'youden']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        optimal_thresholds = {}
        
        for i, metric in enumerate(metrics):
            optimal_thresh, optimal_score, thresholds, scores = self.find_optimal_threshold(
                y_true, y_pred_proba, metric
            )
            optimal_thresholds[metric] = optimal_thresh
            
            axes[i].plot(thresholds, scores, 'b-', linewidth=2)
            axes[i].axvline(x=optimal_thresh, color='red', linestyle='--', 
                          label=f'Optimal: {optimal_thresh:.3f}')
            axes[i].set_xlabel('Threshold')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} vs Threshold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold analysis saved to {save_path}")
        
        plt.show()
        
        return optimal_thresholds
    
    def error_analysis_with_gradcam(self, save_dir, num_samples=10):
        """
        Perform error analysis using Grad-CAM
        """
        print("Performing error analysis with Grad-CAM...")
        
        # Get predictions
        y_pred, y_pred_proba = self.get_predictions(self.X_test)
        
        # Find false positives and false negatives
        false_positives = np.where((y_pred == 1) & (self.y_test == 0))[0]
        false_negatives = np.where((y_pred == 0) & (self.y_test == 1))[0]
        true_positives = np.where((y_pred == 1) & (self.y_test == 1))[0]
        true_negatives = np.where((y_pred == 0) & (self.y_test == 0))[0]
        
        print(f"False Positives: {len(false_positives)}")
        print(f"False Negatives: {len(false_negatives)}")
        print(f"True Positives: {len(true_positives)}")
        print(f"True Negatives: {len(true_negatives)}")
        
        # Create error analysis directory
        error_dir = os.path.join(save_dir, 'error_analysis')
        os.makedirs(error_dir, exist_ok=True)
        
        # Analyze false positives
        if len(false_positives) > 0:
            fp_indices = np.random.choice(false_positives, 
                                        min(num_samples, len(false_positives)), 
                                        replace=False)
            fp_images = self.X_test[fp_indices]
            fp_labels = self.y_test[fp_indices]
            
            print(f"\nAnalyzing {len(fp_indices)} false positives...")
            self.gradcam.generate_multiple_gradcams(
                fp_images, fp_labels,
                save_dir=os.path.join(error_dir, 'false_positives'),
                max_images=len(fp_indices)
            )
        
        # Analyze false negatives
        if len(false_negatives) > 0:
            fn_indices = np.random.choice(false_negatives, 
                                        min(num_samples, len(false_negatives)), 
                                        replace=False)
            fn_images = self.X_test[fn_indices]
            fn_labels = self.y_test[fn_indices]
            
            print(f"\nAnalyzing {len(fn_indices)} false negatives...")
            self.gradcam.generate_multiple_gradcams(
                fn_images, fn_labels,
                save_dir=os.path.join(error_dir, 'false_negatives'),
                max_images=len(fn_indices)
            )
        
        # Analyze some correct predictions for comparison
        tp_indices = np.random.choice(true_positives, 
                                    min(num_samples//2, len(true_positives)), 
                                    replace=False)
        tn_indices = np.random.choice(true_negatives, 
                                    min(num_samples//2, len(true_negatives)), 
                                    replace=False)
        
        correct_indices = np.concatenate([tp_indices, tn_indices])
        correct_images = self.X_test[correct_indices]
        correct_labels = self.y_test[correct_indices]
        
        print(f"\nAnalyzing {len(correct_indices)} correct predictions...")
        self.gradcam.generate_multiple_gradcams(
            correct_images, correct_labels,
            save_dir=os.path.join(error_dir, 'correct_predictions'),
            max_images=len(correct_indices)
        )
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_positives': true_positives,
            'true_negatives': true_negatives
        }
    
    def comprehensive_evaluation(self, save_dir, calibration_method='platt'):
        """
        Perform comprehensive model evaluation
        """
        print("Starting comprehensive evaluation...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Get predictions
        y_pred, y_pred_proba = self.get_predictions(self.X_test)
        
        # Basic metrics
        print("\n" + "="*50)
        print("BASIC METRICS")
        print("="*50)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)  # Recall
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Pneumonia'],
                   yticklabels=['Normal', 'Pneumonia'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # ROC Curve
        print("\n" + "="*50)
        print("ROC ANALYSIS")
        print("="*50)
        
        fpr, tpr, _, auc_score = self.plot_roc_curve(
            self.y_test, y_pred_proba, 
            save_path=os.path.join(save_dir, 'roc_curve.png')
        )
        
        # Precision-Recall Curve
        print("\n" + "="*50)
        print("PRECISION-RECALL ANALYSIS")
        print("="*50)
        
        precision_curve, recall_curve, _, ap_score = self.plot_precision_recall_curve(
            self.y_test, y_pred_proba,
            save_path=os.path.join(save_dir, 'pr_curve.png')
        )
        
        # Calibration Analysis
        print("\n" + "="*50)
        print("CALIBRATION ANALYSIS")
        print("="*50)
        
        # Plot original calibration
        self.plot_calibration_curve(
            self.y_test, y_pred_proba,
            save_path=os.path.join(save_dir, 'calibration_curve_original.png')
        )
        
        # Calibrate probabilities
        calibrator, y_pred_proba_calibrated = self.calibrate_probabilities(calibration_method)
        
        # Plot calibrated probabilities
        self.plot_calibration_curve(
            self.y_test, y_pred_proba_calibrated,
            save_path=os.path.join(save_dir, f'calibration_curve_{calibration_method}.png')
        )
        
        # Threshold Analysis
        print("\n" + "="*50)
        print("THRESHOLD ANALYSIS")
        print("="*50)
        
        optimal_thresholds = self.plot_threshold_analysis(
            self.y_test, y_pred_proba,
            save_path=os.path.join(save_dir, 'threshold_analysis.png')
        )
        
        # Error Analysis with Grad-CAM
        print("\n" + "="*50)
        print("ERROR ANALYSIS WITH GRAD-CAM")
        print("="*50)
        
        error_analysis = self.error_analysis_with_gradcam(save_dir)
        
        # Save comprehensive results
        results = {
            'basic_metrics': {
                'accuracy': float(accuracy),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision),
                'f1_score': float(f1),
                'auc': float(auc),
                'ap_score': float(ap_score)
            },
            'confusion_matrix': cm.tolist(),
            'optimal_thresholds': optimal_thresholds,
            'error_counts': {
                'false_positives': len(error_analysis['false_positives']),
                'false_negatives': len(error_analysis['false_negatives']),
                'true_positives': len(error_analysis['true_positives']),
                'true_negatives': len(error_analysis['true_negatives'])
            },
            'calibration_method': calibration_method
        }
        
        # Save results to JSON
        with open(os.path.join(save_dir, 'comprehensive_evaluation.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save calibrator
        with open(os.path.join(save_dir, f'calibrator_{calibration_method}.pkl'), 'wb') as f:
            pickle.dump(calibrator, f)
        
        print(f"\nComprehensive evaluation completed!")
        print(f"Results saved to {save_dir}")
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/evaluation', help='Output directory')
    parser.add_argument('--calibration_method', type=str, default='platt', 
                       choices=['platt', 'isotonic'], help='Calibration method')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.data_dir)
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(args.output_dir, args.calibration_method)
    
    print("\nEvaluation Summary:")
    print(f"Accuracy: {results['basic_metrics']['accuracy']:.4f}")
    print(f"AUC: {results['basic_metrics']['auc']:.4f}")
    print(f"Sensitivity: {results['basic_metrics']['sensitivity']:.4f}")
    print(f"Specificity: {results['basic_metrics']['specificity']:.4f}")
    print(f"F1-Score: {results['basic_metrics']['f1_score']:.4f}")

if __name__ == "__main__":
    main()