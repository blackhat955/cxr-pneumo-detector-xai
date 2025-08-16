#!/usr/bin/env python3
"""
PyTorch-based CNN model for chest X-ray pneumonia detection.
Compatible with Python 3.13 - no mutex lock issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional, Dict, Any

class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for chest X-ray images.
    """
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        """
        Initialize dataset.
        
        Args:
            images: Array of images (N, H, W, C)
            labels: Array of labels (N,)
            transform: Optional transforms to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image
            image = Image.fromarray(image, mode='RGB')
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Grayscale with channel dimension
            image = image.squeeze(2)
            image = Image.fromarray(image, mode='L')
        else:
            # Grayscale without channel dimension
            image = Image.fromarray(image, mode='L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

class ChestXrayCNN(nn.Module):
    """
    CNN model for chest X-ray pneumonia detection using PyTorch.
    """
    
    def __init__(self, input_shape: Tuple[int, int] = (28, 28), num_classes: int = 2, dropout_rate: float = 0.5):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Input image shape (H, W)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(ChestXrayCNN, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Ultra-simplified convolutional layers for fastest training
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(4, 4)  # Balanced pooling for speed and feature preservation
        self.dropout = nn.Dropout(dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Calculate the size after convolutions
        self._calculate_fc_input_size()
        
        # Ultra-simplified fully connected layers
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def _calculate_fc_input_size(self):
        """
        Calculate the input size for the first fully connected layer.
        """
        # Create a dummy input to calculate the size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *self.input_shape)
            x = self._forward_conv(dummy_input)
            self.fc_input_size = x.view(1, -1).size(1)
    
    def _forward_conv(self, x):
        """
        Forward pass through convolutional layers.
        """
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        return x
    
    def forward(self, x):
        """
        Forward pass through the entire network.
        """
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Get feature maps for Grad-CAM visualization.
        
        Returns:
            Dictionary of feature maps from different layers
        """
        features = {}
        
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        features['conv1'] = x
        x = F.relu(self.bn2(self.conv2(x)))
        features['conv2'] = x
        x = self.pool(x)
        x = self.dropout(x)
        
        # Second conv block
        x = F.relu(self.bn3(self.conv3(x)))
        features['conv3'] = x
        x = F.relu(self.bn4(self.conv4(x)))
        features['conv4'] = x
        x = self.pool(x)
        x = self.dropout(x)
        
        # Third conv block
        x = F.relu(self.bn5(self.conv5(x)))
        features['conv5'] = x
        
        return features

class ModelManager:
    """
    Manager class for PyTorch model operations.
    """
    
    def __init__(self, input_shape: Tuple[int, int] = (28, 28), num_classes: int = 2):
        """
        Initialize model manager.
        
        Args:
            input_shape: Input image shape
            num_classes: Number of classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def create_model(self, dropout_rate: float = 0.5) -> ChestXrayCNN:
        """
        Create and initialize the model.
        
        Args:
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Initialized model
        """
        self.model = ChestXrayCNN(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            dropout_rate=dropout_rate
        )
        self.model.to(self.device)
        
        print(f"✅ Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        print(f"📱 Using device: {self.device}")
        
        return self.model
    
    def create_data_loaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch data loaders.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            batch_size: Batch size
            
        Returns:
            Training and validation data loaders
        """
        train_dataset = ChestXrayDataset(X_train, y_train, transform=self.train_transform)
        val_dataset = ChestXrayDataset(X_val, y_val, transform=self.val_transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        return train_loader, val_loader
    
    def save_model(self, model: nn.Module, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save PyTorch model.
        
        Args:
            model: Model to save
            filepath: Path to save the model
            metadata: Optional metadata to save with model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
        }
        
        if metadata:
            save_dict.update(metadata)
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> ChestXrayCNN:
        """
        Load PyTorch model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Create model with saved parameters
        model = ChestXrayCNN(
            input_shape=checkpoint.get('input_shape', self.input_shape),
            num_classes=checkpoint.get('num_classes', self.num_classes)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"📂 Model loaded from {filepath}")
        return model
    
    def predict(self, model: nn.Module, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction on a single image.
        
        Args:
            model: Trained model
            image: Input image
            
        Returns:
            Predictions and probabilities
        """
        model.eval()
        
        # Preprocess image
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image
            pil_image = Image.fromarray(image.astype('uint8'), mode='RGB')
        else:
            # Grayscale image
            if len(image.shape) == 3:
                image = image.squeeze(-1)  # Remove last dimension if it's 1
            pil_image = Image.fromarray(image.astype('uint8'), mode='L')
        
        tensor_image = self.val_transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = model(tensor_image)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()

def test_pytorch_model():
    """
    Test function to verify PyTorch model works correctly.
    """
    print("Testing PyTorch model...")
    
    try:
        # Create model manager
        manager = ModelManager(input_shape=(28, 28), num_classes=2)
        
        # Create model
        model = manager.create_model()
        
        # Test with dummy data
        dummy_input = torch.randn(1, 1, 28, 28).to(manager.device)
        output = model(dummy_input)
        
        print(f"Model output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Test prediction
        dummy_image = np.random.rand(28, 28, 1)
        pred, prob = manager.predict(model, dummy_image)
        
        print(f"Prediction: {pred}, Probability: {prob}")
        print("PyTorch model test passed!")
        
        return True
        
    except Exception as e:
        print(f"PyTorch model test failed: {e}")
        return False

if __name__ == "__main__":
    test_pytorch_model()