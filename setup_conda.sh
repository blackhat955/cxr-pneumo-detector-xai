#!/bin/bash

# Alternative setup using Conda (faster than compiling Python)
echo "🐍 Setting up Conda environment for PyTorch compatibility..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "📥 Installing Miniconda..."
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3
    source $HOME/miniconda3/bin/activate
    conda init zsh
    echo "🔄 Please restart your terminal and run this script again."
    exit 0
fi

# Remove existing environment if it exists
echo "🗑️  Removing existing conda environment..."
conda env remove -n pneumonia -y 2>/dev/null || true

# Create conda environment with Python 3.11
echo "🏗️  Creating conda environment with Python 3.11..."
conda create -n pneumonia python=3.11 -y

# Activate environment
echo "🔌 Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pneumonia

# Install PyTorch and dependencies
echo "📦 Installing PyTorch and dependencies..."
conda install pytorch torchvision numpy matplotlib scikit-learn jupyter -c pytorch -y
pip install medmnist gradio streamlit torchmetrics albumentations

# Test installation
echo "🧪 Testing PyTorch installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed successfully!')"

# Test the model
echo "🔬 Testing PyTorch model..."
python src/model_pytorch.py

echo "🎉 Conda setup complete! To use:"
echo "   conda activate pneumonia"
echo "   python main_pytorch.py --config quick"