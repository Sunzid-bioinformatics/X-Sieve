FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Run the app
CMD ["streamlit", "run", "app.py"]
