#!/usr/bin/env python3
"""
Quick setup test script to verify the project environment and basic functionality.

This script performs basic checks to ensure:
1. All required packages are installed
2. Directory structure is correct
3. Data preparation works
4. Basic model creation works

Usage:
    python test_setup.py

Author: AI Assistant
Date: 2024
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
sys.path.append(str(src_dir))

def test_imports():
    """
    Test if all required packages can be imported.
    
    Returns:
        bool: True if all imports successful
    """
    print("🔍 Testing package imports...")
    
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow: {tf.__version__}")
        
        import numpy as np
        print(f"  ✅ NumPy: {np.__version__}")
        
        import matplotlib.pyplot as plt
        print(f"  ✅ Matplotlib: {plt.matplotlib.__version__}")
        
        import sklearn
        print(f"  ✅ Scikit-learn: {sklearn.__version__}")
        
        import cv2
        print(f"  ✅ OpenCV: {cv2.__version__}")
        
        import PIL
        print(f"  ✅ Pillow: {PIL.__version__}")
        
        try:
            import gradio as gr
            print(f"  ✅ Gradio: {gr.__version__}")
        except ImportError:
            print("  ⚠️  Gradio not installed (optional)")
        
        try:
            import streamlit as st
            print(f"  ✅ Streamlit: {st.__version__}")
        except ImportError:
            print("  ⚠️  Streamlit not installed (optional)")
        
        try:
            import medmnist
            print(f"  ✅ MedMNIST: {medmnist.__version__}")
        except ImportError:
            print("  ⚠️  MedMNIST not installed - will be needed for data loading")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def test_directory_structure():
    """
    Test if the project directory structure is correct.
    
    Returns:
        bool: True if structure is correct
    """
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "data",
        "data/raw",
        "data/processed",
        "src",
        "notebooks",
        "experiments"
    ]
    
    required_files = [
        "requirements.txt",
        "README.md",
        "main.py",
        "src/prepare_data.py",
        "src/model.py",
        "src/train.py",
        "src/gradcam.py",
        "src/evaluate.py",
        "src/app.py",
        "notebooks/EDA.ipynb"
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"  ✅ Directory: {dir_path}")
        else:
            print(f"  ❌ Missing directory: {dir_path}")
            all_good = False
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"  ✅ File: {file_path}")
        else:
            print(f"  ❌ Missing file: {file_path}")
            all_good = False
    
    return all_good

def test_basic_functionality():
    """
    Test basic functionality of core modules.
    
    Returns:
        bool: True if basic functionality works
    """
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test model creation
        print("  🔧 Testing model creation...")
        from model import create_model
        
        # Create a simple model
        model = create_model(
            model_type="simple_cnn",
            input_shape=(224, 224, 3),
            num_classes=1
        )
        
        if model is not None:
            print(f"    ✅ Simple CNN created successfully")
            print(f"    📊 Model parameters: {model.count_params():,}")
        else:
            print("    ❌ Failed to create model")
            return False
        
        # Test Grad-CAM class
        print("  🔍 Testing Grad-CAM functionality...")
        from gradcam import GradCAM
        
        gradcam = GradCAM(model)
        if gradcam.model is not None:
            print("    ✅ Grad-CAM initialized successfully")
        else:
            print("    ❌ Failed to initialize Grad-CAM")
            return False
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Functionality test error: {e}")
        return False

def test_data_preparation_dry_run():
    """
    Test data preparation without actually downloading data.
    
    Returns:
        bool: True if preparation functions can be imported
    """
    print("\n📊 Testing data preparation (dry run)...")
    
    try:
        from prepare_data import (
            load_pneumonia_mnist,
            preprocess_images,
            create_stratified_splits,
            compute_class_weights
        )
        
        print("  ✅ Data preparation functions imported successfully")
        
        # Test preprocessing with dummy data
        import numpy as np
        dummy_images = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
        dummy_labels = np.random.randint(0, 2, 10)
        
        processed_images = preprocess_images(dummy_images, target_size=(224, 224))
        
        if processed_images.shape == (10, 224, 224, 3):
            print("  ✅ Image preprocessing works correctly")
        else:
            print(f"  ❌ Unexpected processed shape: {processed_images.shape}")
            return False
        
        # Test class weights computation
        class_weights = compute_class_weights(dummy_labels)
        if len(class_weights) == 2:
            print("  ✅ Class weights computation works")
        else:
            print("  ❌ Class weights computation failed")
            return False
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Data preparation test error: {e}")
        return False

def main():
    """
    Run all setup tests.
    """
    print("""
🫁 Chest X-ray Pneumonia Detection - Setup Test
{'='*50}
    """)
    
    tests = [
        ("Package Imports", test_imports),
        ("Directory Structure", test_directory_structure),
        ("Basic Functionality", test_basic_functionality),
        ("Data Preparation", test_data_preparation_dry_run)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 Test Summary:")
    print(f"{'='*50}")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Run: python main.py --quick")
        print("  2. Or: python main.py --production")
        print("  3. Or: python src/prepare_data.py (to start with data only)")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Check file permissions")
        print("  3. Verify Python version (3.8+ recommended)")
    
    print(f"{'='*50}\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())