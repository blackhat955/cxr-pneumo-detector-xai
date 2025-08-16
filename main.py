#!/usr/bin/env python3
"""
Main script to run the complete chest X-ray pneumonia detection pipeline.

This script orchestrates the entire workflow:
1. Data preparation and preprocessing
2. Model training with transfer learning
3. Model evaluation and analysis
4. Optional: Launch interactive demo

Usage:
    python main.py --quick          # Quick baseline with MobileNetV2
    python main.py --production     # Full pipeline with ResNet50
    python main.py --custom --model_type efficientnet_b0 --epochs 20

Author: AI Assistant
Date: 2024
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
sys.path.append(str(src_dir))

def run_command(command, description):
    """
    Run a command and handle errors gracefully.
    
    Args:
        command (list): Command to run as list of strings
        description (str): Description of what the command does
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout[-500:])
        if e.stderr:
            print("Stderr:", e.stderr[-500:])
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_requirements():
    """
    Check if required packages are installed.
    
    Returns:
        bool: True if all requirements are met
    """
    print("🔍 Checking requirements...")
    
    try:
        import tensorflow as tf
        import numpy as np
        import matplotlib.pyplot as plt
        import sklearn
        import cv2
        import PIL
        import medmnist
        
        print(f"✅ TensorFlow: {tf.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ All requirements satisfied!")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def prepare_data():
    """
    Run data preparation script.
    
    Returns:
        bool: True if successful
    """
    return run_command(
        [sys.executable, "src/prepare_data.py"],
        "Preparing and preprocessing data"
    )

def train_model(model_type="resnet", epochs_frozen=10, epochs_finetune=15, batch_size=32):
    """
    Run model training script.
    
    Args:
        model_type (str): Type of model to train
        epochs_frozen (int): Epochs for frozen training
        epochs_finetune (int): Epochs for fine-tuning
        batch_size (int): Batch size for training
    
    Returns:
        tuple: (success, model_path)
    """
    command = [
        sys.executable, "src/train.py",
        "--model_type", model_type,
        "--epochs_frozen", str(epochs_frozen),
        "--epochs_finetune", str(epochs_finetune),
        "--batch_size", str(batch_size)
    ]
    
    success = run_command(command, f"Training {model_type} model")
    
    # Find the latest model in experiments directory
    experiments_dir = Path("experiments")
    if experiments_dir.exists():
        model_dirs = [d for d in experiments_dir.iterdir() if d.is_dir() and model_type in d.name]
        if model_dirs:
            latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
            model_files = list(latest_dir.glob("*_final.h5"))
            if model_files:
                return success, str(model_files[0])
    
    return success, None

def evaluate_model(model_path):
    """
    Run model evaluation script.
    
    Args:
        model_path (str): Path to trained model
    
    Returns:
        bool: True if successful
    """
    if not model_path or not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    return run_command(
        [sys.executable, "src/evaluate.py", "--model_path", model_path],
        "Evaluating model performance"
    )

def launch_demo(model_path, app_type="gradio", port=7860):
    """
    Launch interactive demo application.
    
    Args:
        model_path (str): Path to trained model
        app_type (str): Type of app ('gradio' or 'streamlit')
        port (int): Port number for the app
    
    Returns:
        bool: True if successful
    """
    if not model_path or not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    if app_type == "streamlit":
        command = [
            "streamlit", "run", "src/app.py", "--",
            "--model_path", model_path,
            "--app_type", app_type
        ]
    else:  # gradio
        command = [
            sys.executable, "src/app.py",
            "--model_path", model_path,
            "--app_type", app_type,
            "--port", str(port)
        ]
    
    return run_command(command, f"Launching {app_type} demo application")

def main():
    """
    Main function to orchestrate the complete pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Chest X-ray Pneumonia Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --quick                    # Quick baseline (MobileNetV2, 15 epochs)
  python main.py --production               # Production model (ResNet50, 25 epochs)
  python main.py --custom --model_type efficientnet_b0 --epochs_frozen 8 --epochs_finetune 12
  python main.py --evaluate_only --model_path experiments/model.h5
  python main.py --demo_only --model_path experiments/model.h5 --app_type gradio
        """
    )
    
    # Preset configurations
    parser.add_argument("--quick", action="store_true",
                       help="Quick baseline: MobileNetV2, 5+10 epochs")
    parser.add_argument("--production", action="store_true",
                       help="Production model: ResNet50, 10+15 epochs")
    parser.add_argument("--custom", action="store_true",
                       help="Custom configuration with manual parameters")
    
    # Pipeline control
    parser.add_argument("--skip_data", action="store_true",
                       help="Skip data preparation (use existing processed data)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training (use existing model)")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip evaluation")
    parser.add_argument("--evaluate_only", action="store_true",
                       help="Only run evaluation (requires --model_path)")
    parser.add_argument("--demo_only", action="store_true",
                       help="Only launch demo (requires --model_path)")
    
    # Model parameters
    parser.add_argument("--model_type", default="resnet",
                       choices=["mobilenet", "resnet", "efficientnet_b0", "efficientnet_b3", "simple_cnn"],
                       help="Model architecture to use")
    parser.add_argument("--epochs_frozen", type=int, default=10,
                       help="Number of epochs for frozen training")
    parser.add_argument("--epochs_finetune", type=int, default=15,
                       help="Number of epochs for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    
    # Demo parameters
    parser.add_argument("--model_path", type=str,
                       help="Path to trained model (for evaluation/demo only)")
    parser.add_argument("--app_type", default="gradio",
                       choices=["gradio", "streamlit"],
                       help="Type of demo application")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port for demo application")
    parser.add_argument("--launch_demo", action="store_true",
                       help="Launch demo after training")
    
    args = parser.parse_args()
    
    # Handle preset configurations
    if args.quick:
        args.model_type = "mobilenet"
        args.epochs_frozen = 5
        args.epochs_finetune = 10
        args.batch_size = 32
        print("🚀 Quick baseline configuration selected")
    elif args.production:
        args.model_type = "resnet"
        args.epochs_frozen = 10
        args.epochs_finetune = 15
        args.batch_size = 32
        print("🏭 Production configuration selected")
    elif not args.custom and not args.evaluate_only and not args.demo_only:
        print("⚠️  No configuration specified. Using default production settings.")
        args.model_type = "resnet"
        args.epochs_frozen = 10
        args.epochs_finetune = 15
        args.batch_size = 32
    
    print(f"""
🫁 Chest X-ray Pneumonia Detection Pipeline
{'='*50}
Configuration:
  Model Type: {args.model_type}
  Frozen Epochs: {args.epochs_frozen}
  Fine-tune Epochs: {args.epochs_finetune}
  Batch Size: {args.batch_size}
  Demo App: {args.app_type}
{'='*50}
    """)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Create necessary directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    model_path = args.model_path
    
    try:
        # Demo only mode
        if args.demo_only:
            if not model_path:
                print("❌ --model_path required for demo-only mode")
                return 1
            success = launch_demo(model_path, args.app_type, args.port)
            return 0 if success else 1
        
        # Evaluation only mode
        if args.evaluate_only:
            if not model_path:
                print("❌ --model_path required for evaluation-only mode")
                return 1
            success = evaluate_model(model_path)
            return 0 if success else 1
        
        # Full pipeline
        # Step 1: Data preparation
        if not args.skip_data:
            if not prepare_data():
                print("❌ Data preparation failed")
                return 1
        else:
            print("⏭️  Skipping data preparation")
        
        # Step 2: Model training
        if not args.skip_training:
            success, model_path = train_model(
                args.model_type, 
                args.epochs_frozen, 
                args.epochs_finetune, 
                args.batch_size
            )
            if not success:
                print("❌ Model training failed")
                return 1
        else:
            print("⏭️  Skipping training")
        
        # Step 3: Model evaluation
        if not args.skip_evaluation and model_path:
            if not evaluate_model(model_path):
                print("⚠️  Model evaluation failed, but continuing...")
        elif args.skip_evaluation:
            print("⏭️  Skipping evaluation")
        
        # Step 4: Launch demo (optional)
        if args.launch_demo and model_path:
            print("\n🎉 Training complete! Launching demo...")
            launch_demo(model_path, args.app_type, args.port)
        
        print(f"""
🎉 Pipeline completed successfully!

Next steps:
  1. Review results in experiments/ directory
  2. Launch demo: python main.py --demo_only --model_path {model_path or 'YOUR_MODEL_PATH'}
  3. Run evaluation: python main.py --evaluate_only --model_path {model_path or 'YOUR_MODEL_PATH'}

Model saved at: {model_path or 'Check experiments/ directory'}
        """)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())