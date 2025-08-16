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
                    gradcam_result = self.gradcam_visualizer.visualize_gradcam(
                        processed_image, self.model, target_layer='conv2'
                    )
                    
                    if 'visualization' in gradcam_result:
                        # Convert matplotlib figure to PIL Image
                        buf = io.BytesIO()
                        gradcam_result['visualization'].savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        explanation_img = Image.open(buf)
                        plt.close(gradcam_result['visualization'])
                    
                except Exception as e:
                    print(f"Warning: Could not generate Grad-CAM: {e}")
            
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
            title="Chest X-ray Pneumonia Detection with AI Explainability (PyTorch)",
            description="""
            Upload a chest X-ray image to get an AI prediction for pneumonia detection.
            The system will show both the prediction and a visual explanation of what the AI model focused on.
            
            **MEDICAL DISCLAIMER:**
            This is a demonstration tool for educational purposes only. 
            It should NOT be used for actual medical diagnosis or treatment decisions.
            Always consult qualified healthcare professionals for medical advice.
            """,
            article="""
            ### How it works:
            1. **Deep Learning Model**: Uses a PyTorch CNN trained on chest X-ray images
            2. **Binary Classification**: Distinguishes between Normal and Pneumonia cases
            3. **Grad-CAM Visualization**: Shows which parts of the image influenced the AI's decision
            4. **Confidence Scores**: Provides probability estimates for each class
            
            ### Technical Details:
            - **Framework**: PyTorch with custom CNN architecture
            - **Input**: Grayscale chest X-ray images (any size, auto-resized)
            - **Output**: Binary classification with confidence scores
            - **Explainability**: Gradient-weighted Class Activation Mapping (Grad-CAM)
            - **Training Data**: Medical imaging dataset
            
            ### Limitations:
            - Trained on limited dataset size
            - May not generalize to all X-ray equipment or patient populations
            - Cannot replace professional medical evaluation
            - Should not be used for clinical decision-making
            
            ### Educational Purpose:
            This tool demonstrates the application of AI in medical imaging and the importance
            of explainable AI in healthcare. It shows how deep learning models can assist in
            medical image analysis while emphasizing the critical need for human expertise.
            """,
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

def create_gradio_app(model_path: str, share: bool = False, server_port: int = 7860):
    """
    Create and launch PyTorch Gradio app
    """
    try:
        app = ChestXrayPyTorchApp(model_path)
        interface = app.create_gradio_interface()
        
        print(f"Launching PyTorch Gradio app on port {server_port}...")
        print(f"Using model: {model_path}")
        
        interface.launch(
            share=share,
            server_port=server_port,
            server_name="0.0.0.0",
            show_error=True
        )
        
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
    parser.add_argument('--port', type=int, default=7860,
                       help='Port number for the Gradio app (default: 7860)')
    parser.add_argument('--share', action='store_true',
                       help='Create a public link for the app')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return
    
    # Launch the app
    create_gradio_app(
        model_path=args.model_path,
        share=args.share,
        server_port=args.port
    )

if __name__ == "__main__":
    main()