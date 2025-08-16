import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import medmnist
from medmnist import PneumoniaMNIST
import cv2
from tqdm import tqdm
import pickle

def load_pneumonia_mnist():
    """
    Load PneumoniaMNIST dataset
    """
    print("Loading PneumoniaMNIST dataset...")
    
    # Load train and test sets
    train_dataset = PneumoniaMNIST(split='train', download=True)
    test_dataset = PneumoniaMNIST(split='test', download=True)
    val_dataset = PneumoniaMNIST(split='val', download=True)
    
    # Get data and labels
    train_images, train_labels = train_dataset.imgs, train_dataset.labels.flatten()
    test_images, test_labels = test_dataset.imgs, test_dataset.labels.flatten()
    val_images, val_labels = val_dataset.imgs, val_dataset.labels.flatten()
    
    # Combine train and val for custom split
    all_images = np.concatenate([train_images, val_images], axis=0)
    all_labels = np.concatenate([train_labels, val_labels], axis=0)
    
    print(f"Total images: {len(all_images)}")
    print(f"Test images: {len(test_images)}")
    print(f"Label distribution - Normal: {np.sum(all_labels == 0)}, Pneumonia: {np.sum(all_labels == 1)}")
    
    return all_images, all_labels, test_images, test_labels

def preprocess_images(images, target_size=(224, 224)):
    """
    Preprocess images: resize to target_size and convert to 3-channel
    """
    processed_images = []
    
    for img in tqdm(images, desc="Preprocessing images"):
        # Resize image
        img_resized = cv2.resize(img, target_size)
        
        # Convert grayscale to 3-channel by stacking
        if len(img_resized.shape) == 2:
            img_3channel = np.stack([img_resized] * 3, axis=-1)
        else:
            img_3channel = img_resized
        
        # Normalize to [0, 1]
        img_normalized = img_3channel.astype(np.float32) / 255.0
        
        processed_images.append(img_normalized)
    
    return np.array(processed_images)

def create_data_splits(images, labels, test_size=0.15, val_size=0.15, random_state=42):
    """
    Create stratified train/val/test splits
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_class_weights(y_train):
    """
    Compute class weights for handling imbalanced dataset
    """
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"Class weights: {class_weight_dict}")
    return class_weight_dict

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, class_weights, save_dir):
    """
    Save processed data to disk
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as numpy arrays
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    
    # Save class weights
    with open(os.path.join(save_dir, 'class_weights.pkl'), 'wb') as f:
        pickle.dump(class_weights, f)
    
    print(f"Processed data saved to {save_dir}")

def create_data_arrays(X_train, y_train, X_val, y_val):
    """
    Return data arrays for PyTorch training (no TensorFlow generators)
    """
    print("Preparing data arrays for PyTorch training...")
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    
    # Convert labels to proper format
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    
    return X_train, y_train, X_val, y_val

def main():
    """
    Main function to prepare data
    """
    # Load raw data
    images, labels, test_images, test_labels = load_pneumonia_mnist()
    
    # Preprocess images
    print("Preprocessing training images...")
    processed_images = preprocess_images(images)
    
    print("Preprocessing test images...")
    processed_test_images = preprocess_images(test_images)
    
    # Create data splits
    X_train, X_val, X_test_custom, y_train, y_val, y_test_custom = create_data_splits(
        processed_images, labels
    )
    
    # Use the original test set from PneumoniaMNIST
    X_test = processed_test_images
    y_test = test_labels
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    
    # Save processed data
    save_processed_data(
        X_train, X_val, X_test, y_train, y_val, y_test, 
        class_weights, 'data/processed'
    )
    
    print("Data preparation completed!")
    print(f"Final shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"y_test: {y_test.shape}")

if __name__ == "__main__":
    main()