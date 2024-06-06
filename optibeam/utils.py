import os
import platform, warnings
import numpy as np
import multiprocessing, multiprocess

from skimage.transform import resize
from PIL import Image
from tqdm import tqdm
from functools import wraps, reduce
import time
from typing import *


# ------------------- functional modules -------------------
def add_progress_bar(iterable_arg_index=0):
    """
    Decorator to add a progress bar to the specified iterable argument of a function.
    Parameters:
    - iterable_arg_index (int): The index of the iterable argument in the function's argument list.
    """
    def decorator(func : Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            iterable = args[iterable_arg_index]
            progress_bar = tqdm(iterable)
            new_args = list(args)  
            # Replace the iterable in the arguments with the new progress bar iterator
            new_args[iterable_arg_index] = progress_bar  
            return func(*new_args, **kwargs)
        return wrapper
    return decorator


def combine_functions(functions):
    """
    Combine a list of functions into a single function that processes
    data sequentially through each function in the list.

    Args:
        functions (list[callable]): A list of functions, where each function 
            has the same type of input and output.

    Returns:
        callable: A combined function that is the composition of all the functions 
        in the list. If the input list is empty, returns an identity function.
    """
    if not functions:  
        return lambda x: x
    return reduce(lambda f, g: lambda x: g(f(x)), functions)



def preset_kwargs(**preset_kwargs):
    """
    A decorator to preset keyword arguments of any function. The first argument
    of the function is assumed to be the input data, and the rest are considered
    keyword arguments for tuning or controlling the function's behavior.

    Parameters:
    - **preset_kwargs: Arbitrary keyword arguments that will be preset for the decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Update the preset keyword arguments with explicitly provided ones, if any
            combined_kwargs = {**preset_kwargs, **kwargs}
            return func(*args, **combined_kwargs)
        return wrapper
    return decorator


def timeout(seconds):
    """Decorator to timeout a function after 'seconds' seconds"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)  # Execute the function
            end = time.time()
            if end - start > seconds:
                raise RuntimeError(f"'{func.__name__}' timed out after {seconds} seconds. process terminated.")
            return result
        return wrapper
    return decorator


def deprecated(reason):
    """
    Decorator to mark a function as deprecated.
    """
    def decorator(func):
        @wraps(func)
        def new_func(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn(f"{func.__name__} is deprecated: {reason}",
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return func(*args, **kwargs)
        return new_func
    return decorator



def deprecated_class(reason):
    """
    Decorator to mark a class as deprecated.
    """
    def class_rebuilder(cls):
        orig_init = cls.__init__
        @wraps(cls.__init__)
        def new_init(self, *args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(f"{cls.__name__} is deprecated: {reason}",
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            orig_init(self, *args, **kwargs)
        cls.__init__ = new_init
        return cls
    return class_rebuilder


# ------------------- multiprocessing -------------------

def process_list_in_parallel(function, data_list):
    with multiprocess.Pool(processes=multiprocess.cpu_count()) as pool:
        result = pool.map(function, data_list)
    return result


@deprecated("Not updating anymore, use apply_multiprocess() instead.")
def apply_multiprocessing(function):
    """
    Decorator to apply multiprocessing to a function that processes an iterable.
    No progress indicator, can be used in the terminal.
    """
    @wraps(function)
    def wrapper(iterable):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            result = pool.map(function, iterable)
        return result
    return wrapper


def apply_multiprocess(function):
    """
    Decorator to apply multiprocess to a function that processes an iterable. 
    Use multiprocess which is compatible with Jupyter notebook.
    Could select adding a progress indicator to the operation.
    """
    @wraps(function)
    def wrapper(iterable):
        processor = multiprocess.cpu_count()
        with multiprocess.Pool(processes=processor) as pool:
            print(f"Processing {len(iterable)} items with {processor} processes...")
            result = list(tqdm(pool.imap(function, iterable), total=len(iterable)))
        return result
    return wrapper



# ------------------- file operations -------------------

def get_all_file_paths(dirs, types=['']) -> list:
    """
    Get all file paths in the specified directories with the specified file types.
    input: dirs (list of strings or string of the root of dataset folder), types (list of strings) 
    """
    if isinstance(dirs, str): # Check if dirs is a single string and convert to list if necessary
        dirs = [dirs]
    file_paths = []  
    for dir in dirs:
        for root, _, files in os.walk(dir):
            for file in files:
                if any(type in file for type in types):
                    file_path = os.path.join(root, file)
                    file_paths.append(os.path.abspath(file_path))
    print(f"Found {len(file_paths)} files.")
    return file_paths



class ImageLoader:
    def __init__(self, funcs):
        if not isinstance(funcs, list):
            funcs = [funcs]
        self.funcs = funcs
    
    @deprecated("Use ImageLoader.load instead.")
    def load_image(self, image_path):
        """
        Load an image from the specified path and apply the specified functions to the image sequentially.
        """
        with Image.open(image_path) as img:
            # Convert the image to a NumPy array
            img = np.array(img)
            for func in self.funcs:
                img = func(img)
        return img

    @deprecated("Use ImageLoader.load instead.")
    def load_images(self, image_paths):
        """
        Load multiple images and return a dataset in numpy array format.
        """
        temp = []
        for image_path in image_paths:
            temp.append(self.load_image(image_path))
        dataset = np.array(temp)
        print(f"Loaded dataset shape: {dataset.shape}")
        return dataset
    
    def load(self, input):
        """
        Automatically decide whether to load a single image or multiple images based on the input type.
        """
        if isinstance(input, str):  # Single image path
            return self.load_image(input)
        elif isinstance(input, list):  # Assuming a list of image paths
            return self.load_images(input)
        else:
            raise TypeError("Unsupported input type. Expected a string or a list of strings.")



# ------------------- image processing -------------------

def rgb_to_grayscale(narray_img : np.array):
    """
    input: image in numpy array format
    output: grayscale image in numpy array format by averaging all the colors
    """
    if narray_img.shape[2] == 4:  # If the image has 4 channels (RGBA), ignore the alpha channel.
        narray_img = narray_img[:, :, :3]
    return np.mean(narray_img, axis=2)


def split_image(narray_img : np.array, select='') -> Tuple[np.array, np.array]:
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
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D numpy array.")
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


def scale_image(narray_img, scaling_factor):
    """
    Scales an image by a given scaling factor.
    
    Parameters:
    - image: ndarray, the input image to be scaled.
    - scaling_factor: float, the factor by which the image will be scaled.
    
    Returns:
    - scaled_image: ndarray, the scaled image.
    """
    # Calculate the new dimensions
    new_height = int(narray_img.shape[0] * scaling_factor)
    new_width = int(narray_img.shape[1] * scaling_factor)
    new_dimensions = (new_height, new_width)
    
    # Resize the image
    scaled_image = resize(narray_img, new_dimensions, anti_aliasing=True)
    
    return scaled_image


# ------------------- system/enviornment -------------------

def is_jupyter():
    """Check if Python is running in Jupyter (notebook or lab) or in a command line."""
    try:
        # Attempt to import a Jupyter-specific package
        from IPython import get_ipython
        # If `get_ipython` does not return None, we are in a Jupyter environment
        if get_ipython() is not None:
            return True
    except ImportError:
        # If the import fails, we are not in a Jupyter environment
        pass
    return False


def get_system_info():
    """
    Get system information including the operating system, version, machine, processor, and Python version.
    Returns:
    dict: A dictionary containing the system information.
    """
    system_info = {
        "System": platform.system(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Architecture": platform.architecture()[0],
        "Python Build": platform.python_version()
    }
    return system_info





