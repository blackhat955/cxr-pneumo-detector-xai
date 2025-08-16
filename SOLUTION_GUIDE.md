# TensorFlow/PyTorch Mutex Lock Issue Solution Guide

## Problem Description

You're encountering a `mutex lock failed: Invalid argument` error when trying to run machine learning code with Python 3.13. This is a known compatibility issue between Python 3.13 and several ML libraries including TensorFlow and PyTorch.

## Root Cause

Python 3.13 introduced changes to threading and mutex handling that are incompatible with the current versions of TensorFlow and PyTorch. The error occurs at the C++ level in the underlying libraries.

## Immediate Solutions

### Option 1: Use Python 3.11 or 3.12 (Recommended)

```bash
# Install pyenv to manage Python versions
brew install pyenv

# Install Python 3.11
pyenv install 3.11.9
pyenv local 3.11.9

# Create new virtual environment
python -m venv venv_311
source venv_311/bin/activate

# Install dependencies
pip install -r requirements_pytorch.txt

# Run the project
python main_pytorch.py --config quick
```

### Option 2: Use Conda Environment

```bash
# Install conda if not already installed
brew install --cask miniconda

# Create environment with Python 3.11
conda create -n pneumonia python=3.11 pytorch torchvision numpy matplotlib scikit-learn jupyter
conda activate pneumonia

# Install additional packages
pip install medmnist gradio streamlit

# Run the project
python main_pytorch.py --config quick
```

### Option 3: Use Docker

```bash
# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.11-slim

WORKDIR /app
COPY requirements_pytorch.txt .
RUN pip install -r requirements_pytorch.txt

COPY . .
CMD ["python", "main_pytorch.py", "--config", "quick"]
EOF

# Build and run
docker build -t pneumonia-detection .
docker run -v $(pwd)/experiments:/app/experiments pneumonia-detection
```

### Option 4: Use Google Colab

1. Upload your project files to Google Colab
2. Install dependencies: `!pip install -r requirements_pytorch.txt`
3. Run the pipeline: `!python main_pytorch.py --config quick`

## Alternative: CPU-Only Lightweight Version

If you need to work with Python 3.13, you can use a simplified version that avoids the problematic libraries:

### Install CPU-only PyTorch

```bash
# Install CPU-only PyTorch (might work better with Python 3.13)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Use Scikit-learn Alternative

For immediate testing, you can use a scikit-learn based approach:

```bash
# Install minimal requirements
pip install numpy pandas scikit-learn matplotlib seaborn medmnist

# Run data exploration only
python -c "
from src.prepare_data import DataPreparator
dp = DataPreparator('data/raw', 'data/processed')
dp.download_and_prepare_data()
print('Data preparation completed successfully!')
"
```

## Project Status with Current Setup

### ✅ What Works
- Data preparation and exploration
- Basic NumPy/Pandas operations
- Matplotlib visualizations
- Jupyter notebooks for EDA

### ❌ What Doesn't Work
- TensorFlow model training
- PyTorch model training
- Deep learning inference
- Grad-CAM visualizations

### 🔧 Workarounds Available
- Use classical ML models (Random Forest, SVM) with scikit-learn
- Perform data analysis and visualization
- Develop the pipeline structure
- Test on cloud platforms

## Recommended Next Steps

1. **For Development**: Use Python 3.11 with pyenv
2. **For Production**: Use Docker with Python 3.11
3. **For Experimentation**: Use Google Colab
4. **For Quick Testing**: Use the scikit-learn alternative

## Testing Your Setup

After switching to Python 3.11, test with:

```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} works!')"

# Test the model
python src/model_pytorch.py

# Run quick pipeline
python main_pytorch.py --config quick
```

## Additional Resources

- [Python 3.13 Compatibility Issues](https://github.com/pytorch/pytorch/issues/110436)
- [TensorFlow Python 3.13 Support](https://github.com/tensorflow/tensorflow/issues/62003)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

## Support

If you continue to have issues:
1. Check Python version: `python --version`
2. Check virtual environment: `which python`
3. Try the Docker solution for guaranteed compatibility
4. Use cloud platforms like Google Colab for immediate results