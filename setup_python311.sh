#!/bin/bash

# Setup script for Python 3.11 environment
echo "🔧 Setting up Python 3.11 environment for PyTorch compatibility..."

# Set Python 3.11 as local version for this project
echo "📍 Setting Python 3.11.9 as local version..."
pyenv local 3.11.9

# Verify Python version
echo "✅ Python version:"
python --version

# Remove old virtual environment if it exists
if [ -d "venv_311" ]; then
    echo "🗑️  Removing old virtual environment..."
    rm -rf venv_311
fi

# Create new virtual environment with Python 3.11
echo "🏗️  Creating new virtual environment with Python 3.11..."
python -m venv venv_311

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv_311/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch requirements
echo "📦 Installing PyTorch requirements..."
pip install -r requirements_pytorch.txt

# Test PyTorch installation
echo "🧪 Testing PyTorch installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed successfully!')"

# Test the model
echo "🔬 Testing PyTorch model..."
python src/model_pytorch.py

echo "🎉 Setup complete! You can now run:"
echo "   source venv_311/bin/activate"
echo "   python main_pytorch.py --config quick"