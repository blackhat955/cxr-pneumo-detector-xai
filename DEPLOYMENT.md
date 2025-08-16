# Deployment Guide

This guide explains how to deploy the Chest X-ray Pneumonia Detection app to various cloud platforms.

## 🚀 Deployment Options

### 1. Hugging Face Spaces (Recommended)

**Automatic Deployment via GitHub Actions:**

1. Fork this repository
2. Go to your repository Settings → Secrets and variables → Actions
3. Add the following secrets:
   - `HF_TOKEN`: Your Hugging Face access token
   - `HF_USERNAME`: Your Hugging Face username
4. Push to the `master` branch - the app will automatically deploy!

**Manual Deployment:**

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Choose "Gradio" as the SDK
3. Upload the following files:
   - `src/app_pytorch.py` → `app.py`
   - `src/model_pytorch.py`
   - `src/gradcam_pytorch.py`
   - `experiments/quick_test_model.pth`
   - `requirements_pytorch.txt` → `requirements.txt`

### 2. Railway

1. Connect your GitHub repository to [Railway](https://railway.app)
2. The `railway.json` file will automatically configure the deployment
3. Set environment variables if needed:
   - `PORT`: Will be automatically set by Railway
   - `RAILWAY_ENVIRONMENT`: Automatically set

### 3. Render

1. Connect your GitHub repository to [Render](https://render.com)
2. Create a new Web Service
3. Use the following settings:
   - **Build Command**: `pip install -r requirements_pytorch.txt`
   - **Start Command**: `python src/app_pytorch.py --model_path experiments/quick_test_model.pth --port $PORT --host 0.0.0.0`

### 4. Docker Deployment

**Build and run locally:**
```bash
docker build -t chest-xray-app .
docker run -p 7860:7860 chest-xray-app
```

**Deploy to any Docker-compatible platform:**
- Google Cloud Run
- AWS ECS
- Azure Container Instances
- DigitalOcean App Platform

### 5. Heroku

1. Install Heroku CLI
2. Create a new Heroku app:
```bash
heroku create your-app-name
```
3. Add a `Procfile`:
```
web: python src/app_pytorch.py --model_path experiments/quick_test_model.pth --port $PORT --host 0.0.0.0
```
4. Deploy:
```bash
git push heroku master
```

## 📋 Requirements

- Python 3.11+
- PyTorch
- Gradio
- All dependencies in `requirements_pytorch.txt`

## 🔧 Environment Variables

The app automatically detects deployment environments and configures itself accordingly:

- `PORT`: Server port (default: 7860)
- `HOST`: Server host (default: 127.0.0.1, auto-set to 0.0.0.0 for deployments)
- `RAILWAY_ENVIRONMENT`: Automatically set by Railway
- `RENDER`: Automatically set by Render
- `SPACE_ID`: Automatically set by Hugging Face Spaces

## 🛠️ Troubleshooting

### Common Issues:

1. **Model file not found**: Ensure `experiments/quick_test_model.pth` is included in deployment
2. **Port binding issues**: The app automatically handles port configuration for most platforms
3. **Memory issues**: The model requires ~100MB RAM minimum

### Platform-Specific Notes:

- **Hugging Face Spaces**: Free tier has 16GB storage limit
- **Railway**: Free tier includes 500 hours/month
- **Render**: Free tier spins down after inactivity
- **Heroku**: Free tier discontinued, paid plans available

## 📊 Performance Considerations

- **Cold Start**: First request may take 10-30 seconds
- **Memory Usage**: ~200-500MB depending on platform
- **Response Time**: 2-5 seconds per prediction

## 🔒 Security Notes

- Model predictions are for educational purposes only
- No patient data is stored
- All processing happens server-side
- Consider adding rate limiting for production use

## 📝 Customization

To customize the deployment:

1. Modify `src/app_pytorch.py` for UI changes
2. Update `requirements_pytorch.txt` for dependencies
3. Adjust Docker configuration in `Dockerfile`
4. Configure CI/CD in `.github/workflows/deploy.yml`

---

**Need help?** Check the platform-specific documentation or create an issue in the repository.