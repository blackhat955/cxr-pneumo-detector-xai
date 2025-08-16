#!/usr/bin/env python3
"""
PyTorch-based Grad-CAM implementation for chest X-ray pneumonia detection.
Compatible with Python 3.13 - no mutex lock issues.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import os

from model_pytorch import ChestXrayCNN, ModelManager

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for PyTorch models.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str = 'conv2'):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of the target layer for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.device = next(model.parameters()).device
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register forward and backward hooks for the target layer.
        """
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find the target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")
        
        # Register hooks
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            class_idx: Target class index (if None, use predicted class)
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get the class index
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Generate CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def generate_guided_gradcam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Guided Grad-CAM.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            class_idx: Target class index
            
        Returns:
            Guided Grad-CAM as numpy array
        """
        # Generate regular Grad-CAM
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Generate guided backpropagation
        guided_grads = self._guided_backprop(input_tensor, class_idx)
        
        # Combine CAM and guided gradients
        guided_gradcam = cam[..., np.newaxis] * guided_grads
        
        return guided_gradcam
    
    def _guided_backprop(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate guided backpropagation.
        
        Args:
            input_tensor: Input tensor
            class_idx: Target class index
            
        Returns:
            Guided gradients
        """
        input_tensor.requires_grad_()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get gradients
        gradients = input_tensor.grad.data[0].cpu().numpy()
        
        # Apply guided backpropagation (keep only positive gradients)
        gradients = np.maximum(gradients, 0)
        
        return gradients

class GradCAMVisualizer:
    """
    Visualizer for Grad-CAM results.
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.class_names = ['Normal', 'Pneumonia']
    
    def visualize_gradcam(self, image: np.ndarray, model: torch.nn.Module, 
                         save_path: Optional[str] = None, 
                         target_layer: str = 'conv2') -> Dict[str, Any]:
        """
        Visualize Grad-CAM for an image.
        
        Args:
            image: Input image (H, W) or (H, W, 1)
            model: Trained model
            save_path: Path to save visualization
            target_layer: Target layer for Grad-CAM
            
        Returns:
            Dictionary with visualization results
        """
        # Preprocess image
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to grayscale
            image = np.mean(image, axis=2)
        
        # Ensure image is in [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to PIL and apply transforms
        pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        # Convert grayscale to RGB for the transform
        pil_image_rgb = pil_image.convert('RGB')
        tensor_image = self.model_manager.val_transform(pil_image_rgb).unsqueeze(0)
        tensor_image = tensor_image.to(self.model_manager.device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(tensor_image)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Generate Grad-CAM
        gradcam = GradCAM(model, target_layer)
        
        # Generate heatmaps for both classes
        heatmaps = {}
        for class_idx in range(len(self.class_names)):
            cam = gradcam.generate_cam(tensor_image, class_idx)
            heatmaps[self.class_names[class_idx]] = cam
        
        # Create visualization
        fig = self._create_visualization(
            image, heatmaps, predicted_class, confidence, probabilities[0].cpu().numpy()
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'heatmaps': heatmaps,
            'visualization_path': save_path
        }
    
    def _create_visualization(self, original_image: np.ndarray, heatmaps: Dict[str, np.ndarray],
                            predicted_class: int, confidence: float, probabilities: np.ndarray) -> plt.Figure:
        """
        Create comprehensive Grad-CAM visualization.
        
        Args:
            original_image: Original input image
            heatmaps: Dictionary of class heatmaps
            predicted_class: Predicted class index
            confidence: Prediction confidence
            probabilities: Class probabilities
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Grad-CAM Analysis\nPredicted: {self.class_names[predicted_class]} ({confidence:.2%})', 
                    fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original X-ray')
        axes[0, 0].axis('off')
        
        # Class probability bar chart
        colors = ['lightblue', 'lightcoral']
        bars = axes[0, 1].bar(self.class_names, probabilities, color=colors)
        axes[0, 1].set_title('Class Probabilities')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].set_ylim(0, 1)
        
        # Add probability values on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom')
        
        # Highlight predicted class
        bars[predicted_class].set_color('red' if predicted_class == 1 else 'blue')
        bars[predicted_class].set_alpha(0.8)
        
        # Legend for interpretation
        axes[0, 2].text(0.1, 0.9, 'Grad-CAM Interpretation:', fontsize=12, fontweight='bold', transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.1, 0.8, '• Red areas: High importance', fontsize=10, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.1, 0.7, '• Blue areas: Low importance', fontsize=10, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.1, 0.6, '• Heatmap shows where model', fontsize=10, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.1, 0.5, '  focuses for each class', fontsize=10, transform=axes[0, 2].transAxes)
        axes[0, 2].text(0.1, 0.3, f'Model Confidence: {confidence:.1%}', fontsize=11, fontweight='bold', transform=axes[0, 2].transAxes)
        axes[0, 2].axis('off')
        
        # Grad-CAM heatmaps for each class
        for idx, (class_name, heatmap) in enumerate(heatmaps.items()):
            col = idx
            
            # Resize heatmap to match original image
            heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
            # Create overlay
            overlay = self._create_heatmap_overlay(original_image, heatmap_resized)
            
            axes[1, col].imshow(overlay)
            title = f'{class_name} Grad-CAM'
            if idx == predicted_class:
                title += ' (Predicted)'
            axes[1, col].set_title(title, fontweight='bold' if idx == predicted_class else 'normal')
            axes[1, col].axis('off')
        
        # Combined heatmap (difference)
        if len(heatmaps) == 2:
            normal_heatmap = cv2.resize(heatmaps['Normal'], (original_image.shape[1], original_image.shape[0]))
            pneumonia_heatmap = cv2.resize(heatmaps['Pneumonia'], (original_image.shape[1], original_image.shape[0]))
            
            # Create difference heatmap
            diff_heatmap = pneumonia_heatmap - normal_heatmap
            diff_overlay = self._create_difference_overlay(original_image, diff_heatmap)
            
            axes[1, 2].imshow(diff_overlay)
            axes[1, 2].set_title('Pneumonia vs Normal\n(Red: Pneumonia focus, Blue: Normal focus)')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _create_heatmap_overlay(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """
        Create heatmap overlay on original image.
        
        Args:
            image: Original grayscale image
            heatmap: Grad-CAM heatmap
            alpha: Overlay transparency
            
        Returns:
            RGB overlay image
        """
        # Normalize image to 0-1
        if image.max() > 1:
            image = image / 255.0
        
        # Convert grayscale to RGB
        image_rgb = np.stack([image] * 3, axis=-1)
        
        # Apply colormap to heatmap
        heatmap_colored = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
        
        # Create overlay
        overlay = (1 - alpha) * image_rgb + alpha * heatmap_colored
        
        return np.clip(overlay, 0, 1)
    
    def _create_difference_overlay(self, image: np.ndarray, diff_heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """
        Create difference heatmap overlay.
        
        Args:
            image: Original grayscale image
            diff_heatmap: Difference heatmap (pneumonia - normal)
            alpha: Overlay transparency
            
        Returns:
            RGB overlay image
        """
        # Normalize image
        if image.max() > 1:
            image = image / 255.0
        
        # Convert grayscale to RGB
        image_rgb = np.stack([image] * 3, axis=-1)
        
        # Create custom colormap for difference
        # Positive values (pneumonia focus) -> Red
        # Negative values (normal focus) -> Blue
        diff_colored = np.zeros((*diff_heatmap.shape, 3))
        
        # Normalize difference heatmap
        if np.abs(diff_heatmap).max() > 0:
            diff_norm = diff_heatmap / np.abs(diff_heatmap).max()
        else:
            diff_norm = diff_heatmap
        
        # Red for positive (pneumonia)
        positive_mask = diff_norm > 0
        diff_colored[positive_mask, 0] = diff_norm[positive_mask]  # Red channel
        
        # Blue for negative (normal)
        negative_mask = diff_norm < 0
        diff_colored[negative_mask, 2] = -diff_norm[negative_mask]  # Blue channel
        
        # Create overlay
        overlay = (1 - alpha) * image_rgb + alpha * diff_colored
        
        return np.clip(overlay, 0, 1)
    
    def batch_visualize(self, images: List[np.ndarray], model: torch.nn.Module,
                       save_dir: str = 'experiments/gradcam_results') -> List[Dict[str, Any]]:
        """
        Visualize Grad-CAM for multiple images.
        
        Args:
            images: List of input images
            model: Trained model
            save_dir: Directory to save visualizations
            
        Returns:
            List of visualization results
        """
        os.makedirs(save_dir, exist_ok=True)
        results = []
        
        for i, image in enumerate(images):
            save_path = os.path.join(save_dir, f'gradcam_analysis_{i+1}.png')
            result = self.visualize_gradcam(image, model, save_path)
            results.append(result)
            
            print(f"Image {i+1}: {result['predicted_label']} ({result['confidence']:.2%})")
        
        print(f"Saved {len(images)} Grad-CAM visualizations to {save_dir}")
        return results

def test_gradcam():
    """
    Test Grad-CAM implementation.
    """
    print("Testing PyTorch Grad-CAM...")
    
    try:
        # Create model manager and dummy model
        model_manager = ModelManager(input_shape=(28, 28), num_classes=2)
        model = model_manager.create_model()
        
        # Create dummy image
        dummy_image = np.random.rand(28, 28)
        
        # Create visualizer
        visualizer = GradCAMVisualizer(model_manager)
        
        # Test visualization
        result = visualizer.visualize_gradcam(dummy_image, model)
        
        print(f"Grad-CAM test passed!")
        print(f"Prediction: {result['predicted_label']} ({result['confidence']:.2%})")
        
        return True
        
    except Exception as e:
        print(f"Grad-CAM test failed: {e}")
        return False

if __name__ == "__main__":
    test_gradcam()