import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

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

# def get_all_file_paths(dirs:str, types=['']) -> list:
#     file_paths = []  # List to store file paths
#     for dir in dirs:
#         for root, _, files in os.walk(dir):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 file_paths.append(os.path.abspath(file_path))
#     return [file for file in file_paths if any(type in file for type in types)]


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
    return np.array(temp)



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



def image_normalize(image: np.array):
    """
    Normalize the input image by scaling its pixel values to the range [0, 1].
    Parameters:
    image (np.ndarray): A NumPy array representing the input image.
    Returns:
    np.ndarray: The normalized image.
    """
    return image.astype('float32') / 255.





# ------------------- Quick Plot image -------------------

def plot_narray(image_array, channel=1):    
    """
    Plot a 2D NumPy array as an image.
    Parameters:
    image_array (np.ndarray): A 2D NumPy array to plot as an image.
    """
    # if the image is normalized, convert it back to 0-255 scale
    if np.max(image_array) <= 1:
        image_array = (image_array * 255).astype(np.uint8)
    # Plot the image
    if len(image_array.shape) == 2:
        if channel == 1:
            plt.imshow(image_array, cmap='gray')  # cmap='gray' sets the colormap to grayscale
        else:
            plt.imshow(image_array)
        plt.colorbar()  # Add a color bar to show intensity scale
        plt.title('2D Array Image')  # Add a title
        plt.xlabel('X-axis')  # Label X-axis
        plt.ylabel('Y-axis')  # Label Y-axis
        plt.show()
    else:
        plt.imshow(image_array)
        plt.axis('off')
        plt.show()