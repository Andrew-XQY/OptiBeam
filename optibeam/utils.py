import os, json
import platform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from functools import wraps
from typing import *



# ------------------- progress indicator -------------------
def add_progress_bar(iterable_arg_index=0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            iterable = args[iterable_arg_index]
            progress_bar = tqdm(iterable)
            # Replace the iterable in the arguments with the progress bar
            new_args = list(args)
            new_args[iterable_arg_index] = progress_bar
            return func(*new_args, **kwargs)
        return wrapper
    return decorator


# ------------------- file operations -------------------

def get_all_file_paths(dirs, types=['']) -> list:
    # Check if dirs is a single string and convert to list if necessary
    if isinstance(dirs, str):
        dirs = [dirs]
    file_paths = []  # List to store file paths
    for dir in dirs:
        for root, _, files in os.walk(dir):
            for file in files:
                if any(type in file for type in types):
                    file_path = os.path.join(root, file)
                    file_paths.append(os.path.abspath(file_path))
    print(f"Found {len(file_paths)} files.")
    return file_paths



@add_progress_bar()
def load_images(image_paths, funcs=[]):
    """
    Load an image from the specified paths and apply the specified functions to each image sequentially.
    example: load_images(image_paths, funcs=[np.array, rgb_to_grayscale, split_image, lambda x: x[0].flatten()])
    """
    temp = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            for func in funcs:
                img = func(img)
            temp.append(img)
    dataset = np.array(temp)
    print(f"Loaded dataset shape: {dataset.shape}")
    return dataset



# ------------------- image processing -------------------

def rgb_to_grayscale(narray_img):
    """
    input: image in numpy array format
    output: grayscale image in numpy array format
    """
    if narray_img.shape[2] == 4:  # If the image has 4 channels (RGBA), ignore the alpha channel.
        narray_img = narray_img[:, :, :3]
    return np.mean(narray_img, axis=2) # return grayscale image by averaging all the colors


def split_image(narray_img, select='') -> Tuple[np.array, np.array]:
    """
    input: image in numpy array format
    output: two images, split in the middle horizontally
    """
    left, right = np.array_split(narray_img, 2, axis=1)
    if select not in ['left', 'right']:
        return left, right
    return left if select == 'left' else right


def subtract_minimum(arr):
    """
    Subtract the minimum value from each element in a 1D NumPy array.
    Parameters:
    arr (np.ndarray): A 1D numpy array.
    Returns:
    np.ndarray: The processed array with the minimum value subtracted from each element.
    """
    # Ensure the input is a 1D array
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D numpy array.")
    # Subtract the minimum value from the array
    min_value = np.min(arr)
    processed_arr = arr - min_value
    return processed_arr


def minmax_normalization(arr):
    """
    Min-max normalization
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def image_normalize(narray_img: np.array):
    """
    Normalize the input image by scaling its pixel values to the range [0, 1].
    Parameters:
    image (np.ndarray): A NumPy array representing the input image.
    Returns:
    np.ndarray: The normalized image.
    """
    return narray_img.astype('float32') / 255.





# ------------------- Quick Plot image -------------------

def plot_narray(narray_img, channel=1):    
    """
    Plot a 2D NumPy array as an image.
    Parameters:
    narray_img (np.ndarray): A 2D NumPy array to plot as an image.
    """
    # if the image is normalized, convert it back to 0-255 scale
    if np.max(narray_img) <= 1:
        narray_img = (narray_img * 255).astype(np.uint8)
    # Plot the image
    if len(narray_img.shape) == 2:
        if channel == 1:
            plt.imshow(narray_img, cmap='gray')  # cmap='gray' sets the colormap to grayscale
        else:
            plt.imshow(narray_img)
        plt.colorbar()  # Add a color bar to show intensity scale
        plt.title('2D Array Image')  # Add a title
        plt.xlabel('X-axis')  # Label X-axis
        plt.ylabel('Y-axis')  # Label Y-axis
        plt.show()
    else:
        plt.imshow(narray_img)
        plt.axis('off')
        plt.show()
        


# ------------------- experiment logs -------------------

class Logger:
    """
    Create folder and a log file in the specified directory, containing the experiment details.
    After training, save the log content in the log file under the log directory.
    """
    def __init__(self, log_dir, model=None, dataset=None, history=None, info=''):
        self.log_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_" + info))
        self.model = model
        self.dataset = dataset
        self.history = history
        self.log_file = os.path.join(self.log_dir, 'log.json')
        self.log_content = {'info' : info,
                            'experiment_date' : datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                            'dataset_info': None,
                            'model_info': None, 
                            'training_info': None}
        self.update()
            
    def update(self):
        if self.dataset is not None:
            self.register_dataset()
        if self.model is not None:
            self.register_model()
        if self.history is not None:
            self.register_training()
            
    def register_extra(self, extra_info):
        self.log_content['extra_info'] = extra_info
            
    def register_dataset(self):
        if isinstance(self.dataset, np.ndarray):
            self.log_content['dataset_info'] = {'dataset_shape': str(self.dataset.shape), 
                                                'dataset_dtype': str(self.dataset.dtype),
                                                'dataset_mean': str(np.mean(self.dataset)), 
                                                'dataset_std': str(np.std(self.dataset)),
                                                'dataset_min': str(np.min(self.dataset)), 
                                                'dataset_max': str(np.max(self.dataset))}

    def register_model(self):
        if isinstance(self.model, tf.keras.models.Model):
            self.log_content['model_info'] = self.tf_model_summary()
        
    def register_training(self):
        os_info = {
        "System": platform.system(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Architecture": platform.architecture()[0],
        "Python Build": platform.python_version()
        }
        compiled_info = {
        'loss': self.model.loss,
        'optimizer': type(self.model.optimizer).__name__,
        'optimizer_config': {k:str(v) for k,v in self.model.optimizer.get_config().items()},
        'metrics': [m.name for m in self.model.metrics]
        }
        self.log_content['training_info'] = {'os_info': os_info, 
                                             'compiled_info': compiled_info,
                                             'epoch': len(self.history.epoch),
                                             'training_history': self.history.history
                                             }
        if isinstance(self.model, tf.keras.models.Model):
            compiled_info['tensorflow_version'] = tf.__version__
            
    def tf_model_summary(self):
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        return summary
        
    def log_parse(self):
        pass
        
    def save(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        with open(self.log_file, 'w') as f:
            json.dump(self.log_content, f, indent=4)
        return self.log_file
