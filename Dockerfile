# Use the TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu

# Install any required packages not included in the base image
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0

# Install any required packages not included in the base image
RUN pip install numpy pandas scipy scikit-learn opencv-python matplotlib scikit-image Pillow IPython multiprocess tqdm 

# Copy the entire project directory into /app in the container
COPY . /app

# Set the working directory
WORKDIR /app

# Command to run the main script
# Replace 'main_script.py' with the actual name of your main script
# CMD ["python", "test.py"]
