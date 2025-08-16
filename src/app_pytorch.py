#!/usr/bin/env python3
"""
PyTorch-based Gradio app for chest X-ray pneumonia detection.
Compatible with Python 3.13 - no mutex lock issues.
"""

import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import base64
import os
import argparse
from typing import Tuple, Optional, Dict, Any

# Import PyTorch modules
from model_pytorch import ModelManager, ChestXrayCNN
from gradcam_pytorch import GradCAM, GradCAMVisualizer

class ChestXrayPyTorchApp:
    def __init__(self, model_path: str):
        """
        Initialize the PyTorch-based chest X-ray classification app
        """
        self.model_path = model_path
        self.model = None
        self.model_manager = None
        self.gradcam_visualizer = None
        self.class_names = ['Normal', 'Pneumonia']
        
        self.load_model()
        self.setup_gradcam()
    
    def load_model(self):
        """
        Load the PyTorch model
        """
        try:
            # Create model manager
            self.model_manager = ModelManager(input_shape=(28, 28), num_classes=2)
            
            # Load the trained model
            self.model = self.model_manager.load_model(self.model_path)
            self.model.eval()
            
            print(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def setup_gradcam(self):
        """
        Setup Grad-CAM for explainability
        """
        try:
            self.gradcam_visualizer = GradCAMVisualizer(self.model_manager)
            print("Grad-CAM visualizer initialized")
        except Exception as e:
            print(f"Error setting up Grad-CAM: {e}")
            self.gradcam_visualizer = None
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess the input image for prediction
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed (for medical X-rays)
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=2)
        
        # Normalize to [0, 1]
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        return img_array
    
    def predict_and_explain(self, image: Image.Image) -> Tuple[str, Optional[Image.Image]]:
        """
        Make prediction and generate explanation
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Convert to PIL for transforms
            pil_image = Image.fromarray((processed_image * 255).astype(np.uint8), mode='L')
            pil_image_rgb = pil_image.convert('RGB')
            
            # Apply transforms and make prediction
            tensor_image = self.model_manager.val_transform(pil_image_rgb).unsqueeze(0)
            tensor_image = tensor_image.to(self.model_manager.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(tensor_image)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Generate prediction text
            predicted_label = self.class_names[predicted_class]
            normal_prob = probabilities[0][0].item()
            pneumonia_prob = probabilities[0][1].item()
            
            result_text = f"""
# AI Prediction Results

## Classification
**Prediction:** {predicted_label}  
**Confidence:** {confidence:.1%}

## Probability Breakdown
- **Normal:** {normal_prob:.1%}
- **Pneumonia:** {pneumonia_prob:.1%}

## Model Performance
- **Sensitivity (Recall):** Optimized for detecting pneumonia cases
- **Specificity:** Balanced to minimize false positives

## Important Notes
- This is a **demonstration tool** for educational purposes
- **NOT for clinical diagnosis** or treatment decisions
- Always consult healthcare professionals for medical advice
            """
            
            # Generate Grad-CAM explanation
            explanation_img = None
            if self.gradcam_visualizer:
                try:
                    # Generate Grad-CAM visualization
                    gradcam = GradCAM(self.model, 'conv2')
                    
                    # Apply transforms and make prediction for Grad-CAM
                    tensor_image = self.model_manager.val_transform(pil_image_rgb).unsqueeze(0)
                    tensor_image = tensor_image.to(self.model_manager.device)
                    
                    # Generate CAM for both classes
                    cam_normal_raw = gradcam.generate_cam(tensor_image, 0)  # Normal class
                    cam_pneumonia_raw = gradcam.generate_cam(tensor_image, 1)  # Pneumonia class
                    
                    # Resize heatmaps to match processed image dimensions
                    target_size = processed_image.shape
                    cam_normal = cv2.resize(cam_normal_raw, (target_size[1], target_size[0]))
                    cam_pneumonia = cv2.resize(cam_pneumonia_raw, (target_size[1], target_size[0]))
                    
                    # Create comprehensive medical imaging visualization
                    fig = plt.figure(figsize=(20, 16))
                    
                    # Row 1: Original analysis
                    plt.subplot(4, 5, 1)
                    plt.imshow(processed_image, cmap='gray')
                    plt.title('Original X-ray', fontsize=11, fontweight='bold')
                    plt.axis('off')
                    
                    # Enhanced probability visualization with confidence intervals
                    plt.subplot(4, 5, 2)
                    classes = ['Normal', 'Pneumonia']
                    probs = [normal_prob, pneumonia_prob]
                    colors = ['lightblue', 'red']
                    bars = plt.bar(classes, probs, color=colors, alpha=0.8)
                    plt.title(f'Prediction: {predicted_label}\nConfidence: {confidence:.1%}', fontsize=11, fontweight='bold')
                    plt.ylabel('Probability')
                    plt.ylim(0, 1)
                    for bar, prob in zip(bars, probs):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
                    # Add uncertainty indicator
                    uncertainty = 1 - max(probs)
                    plt.text(0.5, 0.9, f'Uncertainty: {uncertainty:.3f}', ha='center', 
                            transform=plt.gca().transAxes, fontsize=9, 
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
                    
                    # Feature importance histogram
                    plt.subplot(4, 5, 3)
                    feature_importance = np.array([cam_normal.mean(), cam_pneumonia.mean(), 
                                                 cam_normal.max(), cam_pneumonia.max(),
                                                 cam_normal.std(), cam_pneumonia.std()])
                    feature_labels = ['Normal\nMean', 'Pneumonia\nMean', 'Normal\nMax', 
                                    'Pneumonia\nMax', 'Normal\nStd', 'Pneumonia\nStd']
                    colors_feat = ['lightblue', 'red', 'blue', 'darkred', 'cyan', 'orange']
                    plt.bar(range(len(feature_importance)), feature_importance, color=colors_feat, alpha=0.7)
                    plt.title('Feature Activation\nStatistics', fontsize=11, fontweight='bold')
                    plt.xticks(range(len(feature_labels)), feature_labels, rotation=45, fontsize=8)
                    plt.ylabel('Activation')
                    
                    # Attention intensity distribution
                    plt.subplot(4, 5, 4)
                    plt.hist(cam_normal.flatten(), bins=20, alpha=0.6, label='Normal', color='blue', density=True)
                    plt.hist(cam_pneumonia.flatten(), bins=20, alpha=0.6, label='Pneumonia', color='red', density=True)
                    plt.title('Attention Intensity\nDistribution', fontsize=11, fontweight='bold')
                    plt.xlabel('Activation Value')
                    plt.ylabel('Density')
                    plt.legend(fontsize=8)
                    
                    # Model interpretation guide
                    plt.subplot(4, 5, 5)
                    plt.text(0.05, 0.9, 'AI Interpretation Guide:', fontsize=11, fontweight='bold')
                    plt.text(0.05, 0.8, '🔴 Red: High attention areas', fontsize=9)
                    plt.text(0.05, 0.7, '🔵 Blue: Low attention areas', fontsize=9)
                    plt.text(0.05, 0.6, '📊 Statistics show activation patterns', fontsize=9)
                    plt.text(0.05, 0.5, '📈 Distribution shows focus spread', fontsize=9)
                    plt.text(0.05, 0.4, '🎯 Anatomical regions highlighted', fontsize=9)
                    plt.text(0.05, 0.3, '⚡ Uncertainty indicates model doubt', fontsize=9)
                    plt.text(0.05, 0.15, f'🔬 Model Confidence: {confidence:.1%}', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.axis('off')
                    
                    # Row 2: Class-specific Grad-CAMs and analysis
                    plt.subplot(4, 5, 6)
                    plt.imshow(cam_normal, cmap='RdYlBu_r')
                    plt.title('Normal Grad-CAM', fontsize=11, fontweight='bold')
                    plt.axis('off')
                    
                    plt.subplot(4, 5, 7)
                    plt.imshow(cam_pneumonia, cmap='RdYlBu_r')
                    plt.title('Pneumonia Grad-CAM\n(Predicted)', fontsize=11, fontweight='bold')
                    plt.axis('off')
                    
                    # Anatomical region analysis
                    plt.subplot(4, 5, 8)
                    # Divide image into anatomical regions (upper, middle, lower lung)
                    h, w = cam_pneumonia.shape
                    upper_region = cam_pneumonia[:h//3, :].mean()
                    middle_region = cam_pneumonia[h//3:2*h//3, :].mean()
                    lower_region = cam_pneumonia[2*h//3:, :].mean()
                    
                    regions = ['Upper\nLung', 'Middle\nLung', 'Lower\nLung']
                    region_values = [upper_region, middle_region, lower_region]
                    region_colors = ['lightcoral', 'orange', 'lightblue']
                    
                    bars = plt.bar(regions, region_values, color=region_colors, alpha=0.8)
                    plt.title('Anatomical Region\nActivation', fontsize=11, fontweight='bold')
                    plt.ylabel('Mean Activation')
                    for bar, val in zip(bars, region_values):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
                    
                    # Saliency intensity map
                    plt.subplot(4, 5, 9)
                    saliency_map = np.abs(cam_pneumonia - cam_normal)
                    plt.imshow(saliency_map, cmap='hot')
                    plt.title('Differential\nSaliency Map', fontsize=11, fontweight='bold')
                    plt.axis('off')
                    
                    # Comparison view with enhanced visualization
                    plt.subplot(4, 5, 10)
                    comparison = np.zeros((cam_normal.shape[0], cam_normal.shape[1], 3))
                    comparison[:, :, 0] = cam_pneumonia  # Red channel for pneumonia
                    comparison[:, :, 2] = cam_normal     # Blue channel for normal
                    plt.imshow(comparison)
                    plt.title('Class Comparison\n(Red: Pneumonia, Blue: Normal)', fontsize=11, fontweight='bold')
                    plt.axis('off')
                    
                    # Row 3: Advanced overlays and analysis
                    plt.subplot(4, 5, 11)
                    overlay_normal = 0.6 * processed_image + 0.4 * cam_normal
                    plt.imshow(overlay_normal, cmap='gray')
                    plt.title('Normal Focus\nOverlay', fontsize=11, fontweight='bold')
                    plt.axis('off')
                    
                    plt.subplot(4, 5, 12)
                    overlay_pneumonia = 0.6 * processed_image + 0.4 * cam_pneumonia
                    plt.imshow(overlay_pneumonia, cmap='gray')
                    plt.title('Pneumonia Focus\nOverlay', fontsize=11, fontweight='bold')
                    plt.axis('off')
                    
                    # Attention heatmap with contours
                    plt.subplot(4, 5, 13)
                    plt.imshow(processed_image, cmap='gray', alpha=0.7)
                    contours = plt.contour(cam_pneumonia, levels=5, colors='red', alpha=0.8, linewidths=1.5)
                    plt.contour(cam_normal, levels=5, colors='blue', alpha=0.8, linewidths=1.5)
                    plt.title('Attention Contours\n(Red: Pneumonia, Blue: Normal)', fontsize=11, fontweight='bold')
                    plt.axis('off')
                    
                    # Gradient magnitude visualization
                    plt.subplot(4, 5, 14)
                    grad_x = np.gradient(cam_pneumonia, axis=1)
                    grad_y = np.gradient(cam_pneumonia, axis=0)
                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    plt.imshow(gradient_magnitude, cmap='viridis')
                    plt.title('Attention Gradient\nMagnitude', fontsize=11, fontweight='bold')
                    plt.axis('off')
                    
                    # Enhanced comparison overlay with transparency
                    plt.subplot(4, 5, 15)
                    enhanced_overlay = processed_image.copy()
                    enhanced_overlay = np.stack([enhanced_overlay] * 3, axis=-1)
                    enhanced_overlay[:, :, 0] += 0.3 * cam_pneumonia  # Add red for pneumonia
                    enhanced_overlay[:, :, 2] += 0.3 * cam_normal     # Add blue for normal
                    enhanced_overlay = np.clip(enhanced_overlay, 0, 1)
                    plt.imshow(enhanced_overlay)
                    plt.title('Enhanced Multi-Class\nOverlay', fontsize=11, fontweight='bold')
                    plt.axis('off')
                    
                    # Row 4: Statistical analysis and summary
                    plt.subplot(4, 5, 16)
                    # Activation correlation analysis
                    correlation = np.corrcoef(cam_normal.flatten(), cam_pneumonia.flatten())[0, 1]
                    plt.scatter(cam_normal.flatten()[::100], cam_pneumonia.flatten()[::100], 
                               alpha=0.5, s=1, c='purple')
                    plt.xlabel('Normal Activation')
                    plt.ylabel('Pneumonia Activation')
                    plt.title(f'Activation Correlation\nr = {correlation:.3f}', fontsize=11, fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    
                    # Attention focus metrics
                    plt.subplot(4, 5, 17)
                    focus_metrics = {
                        'Normal\nFocus': cam_normal.max() - cam_normal.min(),
                        'Pneumonia\nFocus': cam_pneumonia.max() - cam_pneumonia.min(),
                        'Normal\nSpread': cam_normal.std(),
                        'Pneumonia\nSpread': cam_pneumonia.std()
                    }
                    
                    metric_names = list(focus_metrics.keys())
                    metric_values = list(focus_metrics.values())
                    metric_colors = ['lightblue', 'red', 'blue', 'darkred']
                    
                    plt.bar(range(len(metric_names)), metric_values, color=metric_colors, alpha=0.7)
                    plt.title('Attention Focus\nMetrics', fontsize=11, fontweight='bold')
                    plt.xticks(range(len(metric_names)), metric_names, rotation=45, fontsize=8)
                    plt.ylabel('Value')
                    
                    # Model decision boundary visualization
                    plt.subplot(4, 5, 18)
                    decision_map = cam_pneumonia - cam_normal
                    plt.imshow(decision_map, cmap='RdBu_r', vmin=-1, vmax=1)
                    plt.title('Decision Boundary\n(Red: Pro-Pneumonia)', fontsize=11, fontweight='bold')
                    plt.axis('off')
                    
                    # Uncertainty heatmap
                    plt.subplot(4, 5, 19)
                    uncertainty_map = 1 - np.abs(decision_map)
                    plt.imshow(uncertainty_map, cmap='YlOrRd')
                    plt.title('Uncertainty Map\n(Bright: High Uncertainty)', fontsize=11, fontweight='bold')
                    plt.axis('off')
                    
                    # Summary statistics panel
                    plt.subplot(4, 5, 20)
                    summary_text = f"""DIAGNOSTIC SUMMARY
                    
🔍 Prediction: {predicted_label}
📊 Confidence: {confidence:.1%}
⚠️  Uncertainty: {uncertainty:.3f}

📈 ATTENTION ANALYSIS:
• Normal Focus: {cam_normal.max():.3f}
• Pneumonia Focus: {cam_pneumonia.max():.3f}
• Correlation: {correlation:.3f}

🫁 ANATOMICAL REGIONS:
• Upper Lung: {upper_region:.3f}
• Middle Lung: {middle_region:.3f}
• Lower Lung: {lower_region:.3f}

🎯 KEY FINDINGS:
• Max Attention: {max(cam_pneumonia.max(), cam_normal.max()):.3f}
• Decision Strength: {np.abs(decision_map).max():.3f}"""
                    
                    plt.text(0.05, 0.95, summary_text, fontsize=8, fontfamily='monospace',
                            verticalalignment='top', transform=plt.gca().transAxes,
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.axis('off')
                    
                    plt.tight_layout(pad=2.0)
                    
                    # Convert to PIL Image
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    explanation_img = Image.open(buf)
                    plt.close()
                    
                except Exception as e:
                    print(f"Warning: Could not generate Grad-CAM: {e}")
                    import traceback
                    traceback.print_exc()
            
            return result_text, explanation_img
            
        except Exception as e:
            error_text = f"""
# Error Processing Image

**Error:** {str(e)}

Please try:
1. Uploading a different image
2. Ensuring the image is a valid chest X-ray
3. Checking image format (JPG, PNG supported)
            """
            return error_text, None
    
    def create_gradio_interface(self):
        """
        Create Gradio interface
        """
        def predict_interface(image):
            if image is None:
                return "Please upload an image.", None
            
            result_text, explanation_img = self.predict_and_explain(image)
            return result_text, explanation_img
        
        # Create interface
        interface = gr.Interface(
            fn=predict_interface,
            inputs=[
                gr.Image(type="pil", label="Upload Chest X-ray Image")
            ],
            outputs=[
                gr.Markdown(label="Prediction Results"),
                gr.Image(label="AI Explanation (Grad-CAM)")
            ],
            title="Chest X-ray Pneumonia Detection with AI and Grad-CAM app",
            description="Upload a chest X-ray image to get an AI prediction for pneumonia detection. The system will show both the prediction and a visual explanation of what the AI model focused on.",
            examples=[
                # You can add example images here if available
            ],
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px;
                margin: auto;
            }
            .markdown {
                font-size: 16px;
            }
            """
        )
        
        return interface

def create_gradio_app(model_path: str, share: bool = False, server_port: int = 7860, server_name: str = "127.0.0.1"):
    """
    Create and launch PyTorch Gradio app
    """
    try:
        app = ChestXrayPyTorchApp(model_path)
        interface = app.create_gradio_interface()
        
        print(f"Launching PyTorch Gradio app on {server_name}:{server_port}...")
        print(f"Using model: {model_path}")
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True,
            quiet=False
        )
        
        return interface
        
    except Exception as e:
        print(f"Error launching app: {e}")
        raise e

def main():
    """
    Main function for command line usage
    """
    parser = argparse.ArgumentParser(description='PyTorch Chest X-ray Pneumonia Detection App')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained PyTorch model (.pth file)')
    parser.add_argument('--port', type=int, default=None,
                       help='Port number for the Gradio app (default: 7860)')
    parser.add_argument('--share', action='store_true',
                       help='Create a public link for the app')
    parser.add_argument('--host', type=str, default=None,
                       help='Host to run the app on')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return
    
    # Handle environment variables for deployment
    port = args.port or int(os.environ.get('PORT', 7860))
    host = args.host or os.environ.get('HOST', '127.0.0.1')
    
    # For deployment platforms, use 0.0.0.0
    if os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RENDER') or os.environ.get('SPACE_ID'):
        host = '0.0.0.0'
    
    # Launch the app
    create_gradio_app(
        model_path=args.model_path,
        share=args.share,
        server_port=port,
        server_name=host
    )

if __name__ == "__main__":
    main()