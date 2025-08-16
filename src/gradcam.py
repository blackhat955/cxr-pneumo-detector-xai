import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os

class GradCAM:
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM with a trained model
        
        Args:
            model: Trained Keras model
            layer_name: Name of the layer to generate CAM from (if None, uses last conv layer)
        """
        self.model = model
        self.layer_name = layer_name
        
        if layer_name is None:
            # Find the last convolutional layer
            self.layer_name = self._find_last_conv_layer()
        
        print(f"Using layer '{self.layer_name}' for Grad-CAM")
    
    def _find_last_conv_layer(self):
        """
        Find the last convolutional layer in the model
        """
        for layer in reversed(self.model.layers):
            # Check if layer has 4D output (batch, height, width, channels)
            if len(layer.output_shape) == 4:
                return layer.name
        
        # If no conv layer found, try to find in base model
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'layers'):  # This is likely the base model
                for sublayer in reversed(layer.layers):
                    if len(sublayer.output_shape) == 4:
                        return sublayer.name
        
        raise ValueError("Could not find a suitable convolutional layer")
    
    def generate_heatmap(self, image, class_index=None, alpha=0.4):
        """
        Generate Grad-CAM heatmap for a given image
        
        Args:
            image: Input image (should be preprocessed)
            class_index: Index of the class to generate CAM for (if None, uses predicted class)
            alpha: Transparency for overlay
            
        Returns:
            heatmap: Generated heatmap
            superimposed_img: Original image with heatmap overlay
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Create a model that maps the input image to the activations of the last conv layer
        # as well as the output predictions
        grad_model = keras.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            class_channel = predictions[:, class_index]
        
        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, conv_outputs)
        
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match input image size
        original_image = image[0]
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Create superimposed image
        superimposed_img = self._create_superimposed_image(original_image, heatmap_resized, alpha)
        
        return heatmap_resized, superimposed_img, predictions[0].numpy()
    
    def _create_superimposed_image(self, image, heatmap, alpha=0.4):
        """
        Create superimposed image with heatmap overlay
        """
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to RGB if grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            # If image is already 3-channel, convert to grayscale for better visualization
            image_gray = np.mean(image, axis=2)
            image_rgb = np.stack([image_gray] * 3, axis=2)
        else:
            image_rgb = image
        
        # Apply colormap to heatmap
        heatmap_colored = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
        
        # Superimpose the heatmap on original image
        superimposed_img = heatmap_colored * alpha + image_rgb * (1 - alpha)
        
        return superimposed_img
    
    def visualize_gradcam(self, image, title="Grad-CAM Visualization", save_path=None, 
                         class_index=None, alpha=0.4, figsize=(15, 5)):
        """
        Visualize Grad-CAM results with original image, heatmap, and overlay
        """
        heatmap, superimposed_img, predictions = self.generate_heatmap(image, class_index, alpha)
        
        # Get prediction info
        predicted_prob = predictions[0] if len(predictions.shape) > 0 else predictions
        predicted_class = "Pneumonia" if predicted_prob > 0.5 else "Normal"
        confidence = predicted_prob if predicted_prob > 0.5 else 1 - predicted_prob
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        original_img = image[0] if len(image.shape) == 4 else image
        if original_img.shape[2] == 3:
            # Convert to grayscale for better visualization
            original_display = np.mean(original_img, axis=2)
            axes[0].imshow(original_display, cmap='gray')
        else:
            axes[0].imshow(original_img.squeeze(), cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Superimposed image
        axes[2].imshow(superimposed_img)
        axes[2].set_title(f'Overlay\nPrediction: {predicted_class}\nConfidence: {confidence:.3f}')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to {save_path}")
        
        plt.show()
        
        return heatmap, superimposed_img, predicted_prob
    
    def generate_multiple_gradcams(self, images, labels=None, save_dir=None, 
                                  max_images=9, figsize=(15, 15)):
        """
        Generate Grad-CAM visualizations for multiple images
        """
        n_images = min(len(images), max_images)
        n_cols = 3
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(3, n_cols)
        
        for i in range(n_images):
            image = images[i]
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            heatmap, superimposed_img, predictions = self.generate_heatmap(image)
            
            # Get prediction info
            predicted_prob = predictions[0] if len(predictions.shape) > 0 else predictions
            predicted_class = "Pneumonia" if predicted_prob > 0.5 else "Normal"
            confidence = predicted_prob if predicted_prob > 0.5 else 1 - predicted_prob
            
            # True label if provided
            true_label = ""
            if labels is not None:
                true_class = "Pneumonia" if labels[i] == 1 else "Normal"
                true_label = f"\nTrue: {true_class}"
            
            col = i % n_cols
            row_offset = (i // n_cols) * 3
            
            # Original image
            original_img = image[0]
            if original_img.shape[2] == 3:
                original_display = np.mean(original_img, axis=2)
                axes[row_offset, col].imshow(original_display, cmap='gray')
            else:
                axes[row_offset, col].imshow(original_img.squeeze(), cmap='gray')
            axes[row_offset, col].set_title(f'Original {i+1}{true_label}')
            axes[row_offset, col].axis('off')
            
            # Heatmap
            axes[row_offset + 1, col].imshow(heatmap, cmap='jet')
            axes[row_offset + 1, col].set_title(f'Heatmap {i+1}')
            axes[row_offset + 1, col].axis('off')
            
            # Superimposed
            axes[row_offset + 2, col].imshow(superimposed_img)
            axes[row_offset + 2, col].set_title(f'Overlay {i+1}\nPred: {predicted_class}\nConf: {confidence:.3f}')
            axes[row_offset + 2, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, n_rows * n_cols):
            col = i % n_cols
            row_offset = (i // n_cols) * 3
            for j in range(3):
                axes[row_offset + j, col].axis('off')
        
        plt.suptitle('Grad-CAM Analysis - Multiple Images', fontsize=16)
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'gradcam_multiple_images.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multiple Grad-CAM visualization saved to {save_path}")
        
        plt.show()
    
    def analyze_predictions(self, images, labels, save_dir=None):
        """
        Analyze model predictions with Grad-CAM for correct and incorrect predictions
        """
        correct_predictions = []
        incorrect_predictions = []
        
        for i, (image, label) in enumerate(zip(images, labels)):
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            _, _, predictions = self.generate_heatmap(image)
            predicted_prob = predictions[0] if len(predictions.shape) > 0 else predictions
            predicted_class = 1 if predicted_prob > 0.5 else 0
            
            if predicted_class == label:
                correct_predictions.append((image[0], label, predicted_prob))
            else:
                incorrect_predictions.append((image[0], label, predicted_prob))
        
        print(f"Correct predictions: {len(correct_predictions)}")
        print(f"Incorrect predictions: {len(incorrect_predictions)}")
        
        # Visualize some correct predictions
        if correct_predictions:
            print("\nAnalyzing correct predictions...")
            correct_images = [item[0] for item in correct_predictions[:6]]
            correct_labels = [item[1] for item in correct_predictions[:6]]
            self.generate_multiple_gradcams(
                correct_images, correct_labels, 
                save_dir=os.path.join(save_dir, 'correct_predictions') if save_dir else None,
                max_images=6
            )
        
        # Visualize some incorrect predictions
        if incorrect_predictions:
            print("\nAnalyzing incorrect predictions...")
            incorrect_images = [item[0] for item in incorrect_predictions[:6]]
            incorrect_labels = [item[1] for item in incorrect_predictions[:6]]
            self.generate_multiple_gradcams(
                incorrect_images, incorrect_labels,
                save_dir=os.path.join(save_dir, 'incorrect_predictions') if save_dir else None,
                max_images=6
            )
        
        return correct_predictions, incorrect_predictions

def load_model_and_create_gradcam(model_path, layer_name=None):
    """
    Load a trained model and create GradCAM instance
    """
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    gradcam = GradCAM(model, layer_name)
    
    return model, gradcam

def main():
    """
    Example usage of GradCAM
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/gradcam', help='Output directory')
    parser.add_argument('--layer_name', type=str, default=None, help='Layer name for Grad-CAM')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    # Load model and create GradCAM
    model, gradcam = load_model_and_create_gradcam(args.model_path, args.layer_name)
    
    # Load test data
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze random samples
    indices = np.random.choice(len(X_test), args.num_samples, replace=False)
    sample_images = X_test[indices]
    sample_labels = y_test[indices]
    
    print(f"Analyzing {args.num_samples} random samples...")
    gradcam.generate_multiple_gradcams(
        sample_images, sample_labels, 
        save_dir=args.output_dir, 
        max_images=args.num_samples
    )
    
    # Analyze correct vs incorrect predictions
    print("\nAnalyzing correct vs incorrect predictions...")
    gradcam.analyze_predictions(sample_images, sample_labels, args.output_dir)
    
    print(f"\nGrad-CAM analysis completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()