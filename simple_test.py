#!/usr/bin/env python3
"""
Simple test script to verify basic functionality without TensorFlow.
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """
    Test basic package imports without TensorFlow.
    """
    print("🔍 Testing basic imports...")
    
    try:
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
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_tensorflow_import():
    """
    Test TensorFlow import separately.
    """
    print("\n🧠 Testing TensorFlow import...")
    
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow: {tf.__version__}")
        print(f"  📊 GPU Available: {tf.config.list_physical_devices('GPU')}")
        return True
        
    except ImportError as e:
        print(f"  ❌ TensorFlow import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ TensorFlow error: {e}")
        return False

def test_data_creation():
    """
    Test basic data operations without ML libraries.
    """
    print("\n📊 Testing basic data operations...")
    
    try:
        import numpy as np
        
        # Create dummy data
        dummy_images = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
        dummy_labels = np.random.randint(0, 2, 10)
        
        print(f"  ✅ Created dummy images: {dummy_images.shape}")
        print(f"  ✅ Created dummy labels: {dummy_labels.shape}")
        
        # Test basic preprocessing
        resized = np.repeat(dummy_images[:, :, :, np.newaxis], 3, axis=3)
        print(f"  ✅ Simulated RGB conversion: {resized.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data operations error: {e}")
        return False

def main():
    """
    Run simplified tests.
    """
    print("""
🫁 Simplified Setup Test
{'='*30}
    """)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Operations", test_data_creation),
        ("TensorFlow Import", test_tensorflow_import),
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
    print(f"\n{'='*30}")
    print("📋 Test Summary:")
    print(f"{'='*30}")
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n{'='*30}")
    
    # Check if TensorFlow works
    tf_works = any(name == "TensorFlow Import" and passed for name, passed in results)
    
    if tf_works:
        print("🎉 TensorFlow is working! You can proceed with the full pipeline.")
        print("\nNext steps:")
        print("  python3 main.py --quick")
    else:
        print("⚠️  TensorFlow has issues. This might be due to Python 3.13 compatibility.")
        print("\nRecommendations:")
        print("  1. Try using Python 3.11 or 3.12 instead")
        print("  2. Use conda environment: conda create -n pneumonia python=3.11")
        print("  3. Or try: pip install tensorflow==2.15.0")
    
    return 0

if __name__ == "__main__":
    exit(main())