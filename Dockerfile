# Use the TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu

# Install any required packages not included in the base image
RUN pip install numpy pandas matplotlib sqlite3 PIL scipy cv2 sklearn IPython multiprocess tqdm skimage

# Copy the entire project directory into /app in the container
COPY . /app

# Set the working directory
WORKDIR /app/examples/machine_learning

# Command to run the main script
# Replace 'main_script.py' with the actual name of your main script
CMD ["python", "training.py"]
