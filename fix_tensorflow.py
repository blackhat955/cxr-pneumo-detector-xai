#!/usr/bin/env python3
"""
Script to fix TensorFlow mutex lock issues on Python 3.13.

This script attempts several solutions:
1. Set environment variables to fix threading issues
2. Install compatible TensorFlow version
3. Provide alternative solutions
"""

import os
import sys
import subprocess

def set_tensorflow_env_vars():
    """
    Set environment variables to fix TensorFlow threading issues.
    """
    print("🔧 Setting TensorFlow environment variables...")
    
    # Fix for mutex lock issues
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    os.environ['OMP_NUM_THREADS'] = '1'       # Limit OpenMP threads
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # Limit TensorFlow interop threads
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # Limit TensorFlow intraop threads
    os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Disable GPU to avoid CUDA issues
    
    # Additional fixes for macOS
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    print("  ✅ Environment variables set")

def test_tensorflow_import():
    """
    Test TensorFlow import with environment fixes.
    """
    print("\n🧠 Testing TensorFlow import with fixes...")
    
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow imported successfully: {tf.__version__}")
        
        # Test basic operations
        x = tf.constant([1, 2, 3])
        y = tf.constant([4, 5, 6])
        z = tf.add(x, y)
        print(f"  ✅ Basic operations work: {z.numpy()}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ TensorFlow import failed: {e}")
        return False

def install_compatible_tensorflow():
    """
    Install a more compatible TensorFlow version.
    """
    print("\n📦 Installing compatible TensorFlow version...")
    
    try:
        # Uninstall current TensorFlow
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'tensorflow', '-y'], 
                      check=True, capture_output=True)
        
        # Install TensorFlow 2.15 (more stable with Python 3.13)
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.15.0'], 
                      check=True, capture_output=True)
        
        print("  ✅ TensorFlow 2.15.0 installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Installation failed: {e}")
        return False

def create_tensorflow_test_script():
    """
    Create a standalone test script for TensorFlow.
    """
    print("\n📝 Creating TensorFlow test script...")
    
    test_script = '''
#!/usr/bin/env python3
"""
Standalone TensorFlow test with environment fixes.
"""

import os

# Set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

print("🧠 Testing TensorFlow with environment fixes...")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    # Test basic operations
    print("🔧 Testing basic operations...")
    x = tf.constant([1.0, 2.0, 3.0])
    y = tf.constant([4.0, 5.0, 6.0])
    z = tf.add(x, y)
    print(f"✅ Addition result: {z.numpy()}")
    
    # Test model creation
    print("🏗️ Testing model creation...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print(f"✅ Model created with {model.count_params()} parameters")
    
    print("\n🎉 All TensorFlow tests passed!")
    
except Exception as e:
    print(f"❌ TensorFlow test failed: {e}")
    print("\n💡 Possible solutions:")
    print("  1. Use Python 3.11 or 3.12 instead of 3.13")
    print("  2. Try: conda create -n pneumonia python=3.11 tensorflow")
    print("  3. Use TensorFlow 2.15.0: pip install tensorflow==2.15.0")
    print("  4. Run on Google Colab or cloud environment")
'''
    
    with open('test_tensorflow_fixed.py', 'w') as f:
        f.write(test_script)
    
    print("  ✅ Created test_tensorflow_fixed.py")

def main():
    """
    Main function to fix TensorFlow issues.
    """
    print("""
🔧 TensorFlow Mutex Lock Fix
{'='*30}
    """)
    
    # Step 1: Set environment variables
    set_tensorflow_env_vars()
    
    # Step 2: Test current TensorFlow
    if test_tensorflow_import():
        print("\n🎉 TensorFlow is working! The environment fixes resolved the issue.")
        return True
    
    # Step 3: Try installing compatible version
    print("\n⚠️ Current TensorFlow has issues. Trying compatible version...")
    if install_compatible_tensorflow():
        if test_tensorflow_import():
            print("\n🎉 TensorFlow 2.15.0 is working!")
            return True
    
    # Step 4: Create test script and provide alternatives
    create_tensorflow_test_script()
    
    print("""
❌ TensorFlow mutex lock issue persists.

🔍 Root Cause:
Python 3.13 has threading changes that cause mutex lock issues with TensorFlow.

💡 Recommended Solutions:

1. **Use Python 3.11 or 3.12** (Recommended):
   ```bash
   # Install pyenv to manage Python versions
   brew install pyenv
   pyenv install 3.11.9
   pyenv local 3.11.9
   python -m venv venv_311
   source venv_311/bin/activate
   pip install -r requirements.txt
   ```

2. **Use Conda Environment**:
   ```bash
   conda create -n pneumonia python=3.11 tensorflow keras numpy matplotlib
   conda activate pneumonia
   ```

3. **Use Google Colab**:
   - Upload your project to Google Colab
   - TensorFlow works perfectly there
   - Free GPU access available

4. **Use Docker**:
   ```bash
   docker run -it --rm -v $(pwd):/workspace tensorflow/tensorflow:latest-py3 bash
   ```

5. **Try the test script**:
   ```bash
   python test_tensorflow_fixed.py
   ```

🎯 For now, you can:
- Use the project without TensorFlow for data exploration
- Run notebooks in Jupyter with basic analysis
- Deploy on cloud platforms where TensorFlow works
    """)
    
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)