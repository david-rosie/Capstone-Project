# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Create a .streamlit directory and config optimized for Cloud Run
RUN mkdir -p ~/.streamlit

# Add Streamlit configuration for Cloud Run (faster startup, handles cloud environment)
RUN echo "\
[server]\n\
port = 8080\n\
headless = true\n\
runOnSave = false\n\
enableXsrfProtection = false\n\
maxUploadSize = 200\n\
enableCORS = false\n\
\n\
[client]\n\
showErrorDetails = false\n\
" > ~/.streamlit/config.toml

# Set environment variables for faster startup
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Use exec form to ensure proper signal handling
CMD exec streamlit run frontend_app.py --server.port=8080 --server.address=0.0.0.0
