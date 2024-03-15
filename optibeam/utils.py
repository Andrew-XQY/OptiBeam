import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

from PIL import Image
from tqdm import tqdm
from functools import wraps
from typing import *


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

def get_all_file_paths(dir:str, types=['']) -> list:
    file_paths = []  # List to store file paths
    for root, _, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(os.path.abspath(file_path))
    return [file for file in file_paths if any(type in file for type in types)]


def get_all_images(dir:str, color_space="") -> list:
    paths = get_all_file_paths(dir)
    return [Image.open(i).convert(color_space) if color_space else Image.open(i) for i in tqdm(paths)]


def get_all_images_as_nparray(dir:str, color_space="") -> list:
    return [np.array(i) for i in tqdm(get_all_images(dir, color_space))]




# ------------------- image processing -------------------

def rgb_to_grayscale(img):
    if img.shape[2] == 4:  # If the image has 4 channels (RGBA), ignore the alpha channel.
        img = img[:, :, :3]
    return np.mean(img, axis=2) # return grayscale image by averaging all the colors


def split_image(narray_img) -> Tuple[np.array, np.array]:
    """
    input: grayscale image in numpy array format
    output: two images, split in the middle horizontally
    """
    return np.array_split(narray_img, 2, axis=1)


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