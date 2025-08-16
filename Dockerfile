FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_pytorch.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_pytorch.txt

# Copy source code
COPY src/ ./src/
COPY experiments/ ./experiments/

# Expose port
EXPOSE 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "src/app_pytorch.py", "--model_path", "experiments/quick_test_model.pth", "--port", "7860"]