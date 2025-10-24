# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and code
COPY requirements.txt .
COPY ml_model.pkl .
COPY ml_service.py .
COPY static/ ./static/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Start FastAPI server
CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "5000"]
