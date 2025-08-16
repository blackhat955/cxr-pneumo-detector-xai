#!/usr/bin/env python3
"""
Deploy PyTorch app to Hugging Face Spaces
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def create_hf_space():
    """
    Create and deploy to Hugging Face Spaces
    """
    # Configuration
    space_name = "chest-xray-pneumonia-detector"
    username = os.getenv('HF_USERNAME', 'devil66')  # Set this in GitHub secrets
    token = os.getenv('HF_TOKEN')
    
    if not token:
        print("HF_TOKEN not found in environment variables")
        return
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Create space repository
    repo_id = f"{username}/{space_name}"
    
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            token=token,
            exist_ok=True
        )
        print(f"Space created/updated: {repo_id}")
    except Exception as e:
        print(f"Error creating space: {e}")
        return
    
    # Prepare files for upload
    files_to_upload = [
        ("src/app_pytorch.py", "app_pytorch.py"),
        ("src/model_pytorch.py", "model_pytorch.py"), 
        ("src/gradcam_pytorch.py", "gradcam_pytorch.py"),
        ("experiments/quick_test_model.pth", "quick_test_model.pth"),
        ("requirements_hf.txt", "requirements.txt")
    ]
    
    # Create app.py for Hugging Face Spaces
    app_content = '''#!/usr/bin/env python3
"""
Hugging Face Spaces deployment for Chest X-ray Pneumonia Detection
"""

import os
import gradio as gr
from app_pytorch import ChestXrayPyTorchApp

def main():
    # Use the model from the uploaded file
    model_path = "quick_test_model.pth"
    
    # Create app instance
    app = ChestXrayPyTorchApp(model_path)
    interface = app.create_gradio_interface()
    
    # Launch with proper Hugging Face Spaces configuration
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
'''
    
    with open("app.py", "w") as f:
        f.write(app_content)
    
    # Upload files
    try:
        # Upload main app file
        api.upload_file(
            path_or_fileobj="app.py",
            path_in_repo="app.py",
            repo_id=repo_id,
            repo_type="space",
            token=token
        )
        
        # Upload source files
        for source_path, target_name in files_to_upload:
            if os.path.exists(source_path):
                api.upload_file(
                    path_or_fileobj=source_path,
                    path_in_repo=target_name,
                    repo_id=repo_id,
                    repo_type="space",
                    token=token
                )
                print(f"Uploaded: {source_path} -> {target_name}")
            else:
                print(f"File not found: {source_path}")
        
        # Create README for the space
        readme_content = '''---
title: Chest X-ray Pneumonia Detection with AI and Grad-CAM
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Chest X-ray Pneumonia Detection with AI and Grad-CAM

An AI-powered application for detecting pneumonia in chest X-ray images using PyTorch and Grad-CAM explanations.

## Features
- Deep learning model for pneumonia detection
- Grad-CAM visualizations for AI explainability
- Interactive web interface built with Gradio

## Usage
1. Upload a chest X-ray image
2. Get AI prediction with confidence score
3. View Grad-CAM explanation showing model focus areas

**Disclaimer**: This is a demonstration tool for educational purposes only. Not for clinical diagnosis.
'''
        
        with open("README.md", "w") as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
            token=token
        )
        
        print(f"\n✅ Deployment successful!")
        print(f"🚀 Your app is available at: https://huggingface.co/spaces/{repo_id}")
        
    except Exception as e:
        print(f"Error uploading files: {e}")
    
    # Cleanup
    if os.path.exists("app.py"):
        os.remove("app.py")
    if os.path.exists("README.md"):
        os.remove("README.md")

if __name__ == "__main__":
    create_hf_space()