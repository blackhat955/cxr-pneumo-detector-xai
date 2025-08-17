import gradio as gr
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import base64
import os
import pickle
from gradcam import GradCAM

class ChestXrayApp:
    def __init__(self, model_path, calibrator_path=None):
        """
        Initialize the chest X-ray classification app
        """
        self.model_path = model_path
        self.calibrator_path = calibrator_path
        self.model = None
        self.calibrator = None
        self.gradcam = None
        
        self.load_model()
        if calibrator_path and os.path.exists(calibrator_path):
            self.load_calibrator()
        
        self.setup_gradcam()
    
    def load_model(self):
        """
        Load the trained model
        """
        print(f"Loading model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
    
    def load_calibrator(self):
        """
        Load probability calibrator
        """
        print(f"Loading calibrator from {self.calibrator_path}...")
        with open(self.calibrator_path, 'rb') as f:
            self.calibrator = pickle.load(f)
        print("Calibrator loaded successfully!")
    
    def setup_gradcam(self):
        """
        Setup Grad-CAM for explainability
        """
        self.gradcam = GradCAM(self.model)
        print("Grad-CAM setup completed!")
    
    def preprocess_image(self, image):
        """
        Preprocess uploaded image for model prediction
        """
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        
        # Resize to model input size
        image_resized = cv2.resize(image, (224, 224))
        
        # Convert to 3-channel by stacking
        image_3channel = np.stack([image_resized] * 3, axis=-1)
        
        # Normalize to [0, 1]
        image_normalized = image_3channel.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch, image_normalized
    
    def predict_and_explain(self, image):
        """
        Make prediction and generate Grad-CAM explanation
        """
        try:
            # Preprocess image
            image_batch, image_normalized = self.preprocess_image(image)
            
            # Make prediction
            prediction_proba = self.model.predict(image_batch, verbose=0)[0][0]
            
            # Apply calibration if available
            if self.calibrator is not None:
                prediction_proba_calibrated = self.calibrator.predict_proba(
                    prediction_proba.reshape(-1, 1)
                )[0][1]
            else:
                prediction_proba_calibrated = prediction_proba
            
            # Determine prediction
            prediction_class = "Pneumonia" if prediction_proba_calibrated > 0.5 else "Normal"
            confidence = prediction_proba_calibrated if prediction_proba_calibrated > 0.5 else 1 - prediction_proba_calibrated
            
            # Generate Grad-CAM
            heatmap, superimposed_img, _ = self.gradcam.generate_heatmap(image_batch)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image_normalized[:, :, 0], cmap='gray')
            axes[0].set_title('Original X-ray')
            axes[0].axis('off')
            
            # Heatmap
            im = axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Attention Heatmap')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Superimposed
            axes[2].imshow(superimposed_img)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            # Convert to PIL Image
            explanation_img = Image.open(buf)
            
            # Prepare result text
            result_text = f"""
            **Prediction: {prediction_class}**
            
            **Confidence: {confidence:.1%}**
            
            **Raw Probability: {prediction_proba:.3f}**
            
            {f"**Calibrated Probability: {prediction_proba_calibrated:.3f}**" if self.calibrator else ""}
            
            ---
            
            **Interpretation:**
            The heatmap shows which areas of the X-ray the AI model focused on when making its prediction. 
            Red/yellow areas indicate regions that strongly influenced the decision, while blue areas had less impact.
            
            **Important Note:**
            This is an AI demonstration tool and should NOT be used for actual medical diagnosis. 
            Always consult qualified healthcare professionals for medical advice.
            """
            
            return result_text, explanation_img
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            return error_msg, None
    
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
                gr.Image(type="pil", label="Upload Chest X-ray Image", sources=["upload"])
            ],
            outputs=[
                gr.Markdown(label="Prediction Results"),
                gr.Image(label="AI Explanation (Grad-CAM)")
            ],
            title="🫁 Chest X-ray Pneumonia Detection with AI Explainability",
            description="""
            Upload a chest X-ray image to get an AI prediction for pneumonia detection.
            The system will show both the prediction and a visual explanation of what the AI model focused on.
            
            **⚠️ MEDICAL DISCLAIMER:**
            This is a demonstration tool for educational purposes only. 
            It should NOT be used for actual medical diagnosis or treatment decisions.
            Always consult qualified healthcare professionals for medical advice.
            """,
            article="""
            ### How it works:
            1. **Deep Learning Model**: Uses a convolutional neural network trained on chest X-ray images
            2. **Transfer Learning**: Built on pre-trained models (ResNet50/MobileNetV2) for better performance
            3. **Grad-CAM Visualization**: Shows which parts of the image influenced the AI's decision
            4. **Probability Calibration**: Provides more reliable confidence estimates
            
            ### Technical Details:
            - **Input**: 224x224 pixel chest X-ray images
            - **Output**: Binary classification (Normal vs Pneumonia)
            - **Explainability**: Gradient-weighted Class Activation Mapping (Grad-CAM)
            - **Training Data**: PneumoniaMNIST dataset
            
            ### Limitations:
            - Trained on limited dataset size
            - May not generalize to all X-ray equipment or patient populations
            - Cannot replace professional medical evaluation
            - Should not be used for clinical decision-making
            """,
            examples=[
                # You can add example images here
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
    
    def create_streamlit_app(self):
        """
        Create Streamlit app
        """
        st.set_page_config(
            page_title="Chest X-ray Pneumonia Detection",
            page_icon="🫁",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Title and description
        st.title("🫁 Chest X-ray Pneumonia Detection with AI Explainability")
        
        st.markdown("""
        Upload a chest X-ray image to get an AI prediction for pneumonia detection.
        The system will show both the prediction and a visual explanation of what the AI model focused on.
        """)
        
        # Medical disclaimer
        st.error("""
        ⚠️ **MEDICAL DISCLAIMER**: This is a demonstration tool for educational purposes only. 
        It should NOT be used for actual medical diagnosis or treatment decisions.
        Always consult qualified healthcare professionals for medical advice.
        """)
        
        # Sidebar with information
        with st.sidebar:
            st.header("ℹ️ About This Tool")
            st.markdown("""
            ### How it works:
            - **Deep Learning**: CNN trained on chest X-rays
            - **Transfer Learning**: Pre-trained backbone
            - **Grad-CAM**: Visual explanations
            - **Calibration**: Reliable probabilities
            
            ### Technical Details:
            - Input: 224x224 pixel images
            - Output: Normal vs Pneumonia
            - Training: PneumoniaMNIST dataset
            
            ### Limitations:
            - Limited training data
            - Not for clinical use
            - Requires professional evaluation
            """)
        
        # Main interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a chest X-ray image...",
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                help="Upload a chest X-ray image in common formats (PNG, JPG, etc.)"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded X-ray", use_column_width=True)
                
                # Predict button
                if st.button("🔍 Analyze X-ray", type="primary"):
                    with st.spinner("Analyzing image..."):
                        result_text, explanation_img = self.predict_and_explain(image)
                    
                    # Store results in session state
                    st.session_state.result_text = result_text
                    st.session_state.explanation_img = explanation_img
        
        with col2:
            st.header("📊 Results")
            
            # Display results if available
            if hasattr(st.session_state, 'result_text') and st.session_state.result_text:
                if st.session_state.explanation_img is not None:
                    # Display prediction results
                    st.markdown(st.session_state.result_text)
                    
                    # Display explanation image
                    st.image(
                        st.session_state.explanation_img, 
                        caption="AI Explanation (Grad-CAM)",
                        use_column_width=True
                    )
                else:
                    st.error(st.session_state.result_text)
            else:
                st.info("Upload an image and click 'Analyze X-ray' to see results.")
        
        # Additional information
        with st.expander("📚 Learn More About the Technology"):
            st.markdown("""
            ### Grad-CAM (Gradient-weighted Class Activation Mapping)
            
            Grad-CAM is a technique that helps us understand what parts of an image a deep learning model 
            is focusing on when making predictions. It works by:
            
            1. **Computing gradients** of the predicted class with respect to feature maps
            2. **Weighting feature maps** by the importance of each feature
            3. **Creating a heatmap** that highlights important regions
            4. **Overlaying the heatmap** on the original image
            
            ### Model Architecture
            
            Our model uses transfer learning with a pre-trained backbone (ResNet50 or MobileNetV2) 
            that has been fine-tuned on chest X-ray data. The architecture includes:
            
            - **Backbone**: Pre-trained CNN for feature extraction
            - **Global Average Pooling**: Reduces spatial dimensions
            - **Dropout**: Prevents overfitting
            - **Dense Layer**: Final classification layer
            
            ### Training Process
            
            1. **Data Preprocessing**: Images resized to 224x224, normalized
            2. **Data Augmentation**: Rotation, translation, zoom for robustness
            3. **Transfer Learning**: Frozen backbone → Fine-tuning
            4. **Class Balancing**: Weighted loss for imbalanced data
            5. **Calibration**: Post-training probability calibration
            """)

def create_gradio_app(model_path, calibrator_path=None, share=False, server_port=7860):
    """
    Create and launch Gradio app
    """
    app = ChestXrayApp(model_path, calibrator_path)
    interface = app.create_gradio_interface()
    
    print(f"Launching Gradio app on port {server_port}...")
    interface.launch(
        share=share,
        server_port=server_port,
        server_name="0.0.0.0",
        show_error=True
    )

def create_streamlit_app(model_path, calibrator_path=None):
    """
    Create Streamlit app
    """
    # Initialize app in session state to avoid reloading
    if 'app' not in st.session_state:
        st.session_state.app = ChestXrayApp(model_path, calibrator_path)
    
    # Create the app interface
    st.session_state.app.create_streamlit_app()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch chest X-ray classification app')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--calibrator_path', type=str, default=None, help='Path to calibrator pickle file')
    parser.add_argument('--app_type', type=str, default='gradio', choices=['gradio', 'streamlit'],
                       help='Type of app to launch')
    parser.add_argument('--port', type=int, default=7860, help='Port for Gradio app')
    parser.add_argument('--share', action='store_true', help='Create public link for Gradio app')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    if args.calibrator_path and not os.path.exists(args.calibrator_path):
        print(f"Warning: Calibrator file not found at {args.calibrator_path}")
        args.calibrator_path = None
    
    if args.app_type == 'gradio':
        create_gradio_app(
            model_path=args.model_path,
            calibrator_path=args.calibrator_path,
            share=args.share,
            server_port=args.port
        )
    elif args.app_type == 'streamlit':
        print("To run Streamlit app, use: streamlit run app.py -- --model_path <path> [--calibrator_path <path>]")
        create_streamlit_app(
            model_path=args.model_path,
            calibrator_path=args.calibrator_path
        )

if __name__ == "__main__":
    main()