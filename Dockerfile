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

# Create a .streamlit directory and config
RUN mkdir -p ~/.streamlit

# Add Streamlit configuration for Docker
RUN echo "\
[server]\n\
port = 8501\n\
headless = true\n\
runOnSave = true\n\
" > ~/.streamlit/config.toml

# Expose port 8501 (Streamlit default)
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "frontend_app.py"]
