FROM tensorflow/tensorflow:latest-gpu

# Install additional Python packages
RUN pip install numpy pandas matplotlib

# Copy your application files
COPY . /app

WORKDIR /app
CMD ["python", "your_script.py"]
