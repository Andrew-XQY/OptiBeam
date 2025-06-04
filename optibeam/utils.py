import os, sys
import gc
import cv2
import numpy as np
import threading
import platform, warnings
import multiprocessing, multiprocess, subprocess
import shutil
import random
import tarfile
from skimage.transform import resize
from functools import wraps, reduce
from PIL import Image
from tqdm import tqdm
from typing import *



# ------------------- functional modules -------------------
def identity(x):
    """
    The identity function that returns the input as output.
    """
    return x

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
    
    Returns:
    - function: The decorated function with the preset keyword arguments. (closure)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Update the preset keyword arguments with explicitly provided ones, if any
            combined_kwargs = {**preset_kwargs, **kwargs}
            return func(*args, **combined_kwargs)
        return wrapper
    return decorator


class TimeoutError(Exception):
    pass


def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]  # Placeholder for the function's result
            def function_thread():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=function_thread)
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                thread.join(0.1)  # Ensure any remaining operations wrap up
                if isinstance(result[0], Exception):
                    raise result[0]
                raise TimeoutError("Function call timed out")
            return result[0]
        return wrapper
    return decorator

def print_underscore(func):
    """
    Decorator to print a line of underscores before and after the decorated function is called.
    """
    def wrapper(*args, **kwargs):
        print("-" * 80)
        result = func(*args, **kwargs)
        print("-" * 80)
        return result
    return wrapper


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

def list_to_generator(lst):
    for item in lst:
        yield item

def select_random_elements(lst, n):
    """
    Randomly select n elements from a list.

    Args:
        lst (list): The original list to select elements from.
        n (int): The number of elements to select.

    Returns:
        list: A sublist containing n randomly selected elements.

    Raises:
        ValueError: If n is greater than the length of the list.
    """
    if n > len(lst):
        raise ValueError("n cannot be greater than the length of the list")
    
    return random.sample(lst, n)

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


# ----------------- numerical operations ----------------
def remap_array(array: np.array, new_min: float, new_max: float,
                old_min: float=None, old_max: float=None) -> np.array:
    """
    Remap the elements of a NumPy array from its current range to a new specified range.
    
    Args:
        values (np.array): Original array of values.
        new_min (float): Minimum value of the new range.
        new_max (float): Maximum value of the new range.
    
    Returns:
        np.array: Array with values remapped to the new range.
    """
    if old_min or old_max is None:
        old_min = np.min(array)
        old_max = np.max(array)
    # Avoid division by zero if the array is constant
    if old_min == old_max:
        return np.full_like(array, new_min)
    # Scale the array from the old range to [0, 1], then to [new_min, new_max]
    return new_min + ((array - old_min) * (new_max - new_min) / (old_max - old_min))


# ------------------- file operations -------------------

def count_files_in_directory(directory: str, file_types: list = None, recursive: bool = True) -> int:
    """
    Counts the total number of files in the specified directory based on file types.

    Args:
        directory (str): Path to the directory.
        file_types (list): List of file extensions to filter by (e.g., ['pdf', 'png']).
                         If None or empty, counts all files.
        recursive (bool): Whether to include subdirectories. Default is True.

    Returns:
        int: Total number of files in the directory.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The provided path is not a valid directory: {directory}")

    total_files = 0

    if file_types:
        # Ensure all file types are lowercase for case-insensitive matching
        file_types = [ext.lower() for ext in file_types]

    if recursive:
        # Walk through the directory tree
        for root, _, files in os.walk(directory):
            for file in files:
                if not file_types or file.lower().split('.')[-1] in file_types:
                    total_files += 1
    else:
        # Only look at the top-level directory
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                if not file_types or file.lower().split('.')[-1] in file_types:
                    total_files += 1

    return total_files


def get_all_file_paths(
    dirs: Union[str, List[str]],
    types: Optional[List[str]] = None
) -> List[str]:
    """
    If `types` is None or empty → return every file under `dirs`.
    Otherwise → return only files whose filename contains at least one of the substrings in `types`.
    """
    if isinstance(dirs, str):
        dirs = [dirs]

    # Ensure types is a list/tuple if provided. If it’s None or empty, we'll skip filtering.
    if types is not None and not isinstance(types, (list, tuple)):
        types = [types]

    file_paths = []
    for d in dirs:
        for root, _, files in os.walk(d):
            for fname in files:
                # If types is None or an empty list → not types is True → accept everything.
                # Otherwise → only accept if any(t in fname for t in types).
                if not types or any(t in fname for t in types):
                    file_paths.append(os.path.abspath(os.path.join(root, fname)))
                    
    print(f"Found {len(file_paths)} files.")
    return file_paths


def load_image_as_narray(image_path):
    """
    (Newest method to load image as numpy array)
    Load an image from the specified path and convert it to a NumPy array.
    Includes checks for successful loading, correct data types, and expected pixel range.
    
    Args:
        image_path (str): The file path to the image.
    
    Returns:
        numpy.ndarray: The image as a NumPy array.
    """
    try:
        # Attempt to open the image file
        img = Image.open(image_path)
        # Convert the image to a NumPy array
        img_array = np.array(img)
        # Check if the image array is empty
        if img_array.size == 0:
            raise ValueError("Image is empty")
        # Check data type and range
        if img_array.dtype != 'uint8':
            raise TypeError("Unexpected data type; expected uint8")
        if img_array.min() < 0 or img_array.max() > 255:
            raise ValueError("Pixel values out of expected range (0-255)")
        return img_array

    except FileNotFoundError:
        raise FileNotFoundError("File not found. Please check the file path.")
    except IOError:
        raise IOError("Error in loading the image. The file may be corrupted or unsupported.")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


class ImageLoader:
    def __init__(self, funcs: list=[]):
        if not isinstance(funcs, list):
            funcs = [funcs]
        self.funcs = funcs
    
    #@deprecated("Use ImageLoader.load instead.")
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

    #@deprecated("Use ImageLoader.load instead.")
    def load_images(self, image_paths):
        """
        Load multiple images and return a dataset in numpy array format.
        """
        dataset = []
        for image_path in image_paths:
            dataset.append(self.load_image(image_path))
        print(f"Loaded dataset length: {len(dataset)}")
        gc.collect()  # Optionally collect garbage after each major transformation
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
        

def is_file_locked(file_path):
    """
    Checks if a file is locked.
    For Windows, assumes no lock exists.
    For Linux, uses the `lsof` command to detect locks.
    
    Args:
        file_path (str): Path to the file to check.
    
    Returns:
        bool: True if the file is locked, False otherwise.
    """
    if platform.system() == "Windows":
        # Windows: Assume no lock (built-in limitations for lock checking)
        return False
    elif platform.system() == "Linux":
        # Linux: Check with `lsof`
        try:
            result = subprocess.run(
                ['lsof', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return bool(result.stdout.strip())  # Locked if output exists
        except FileNotFoundError:
            print("Error: `lsof` command not found. Install it on your system.")
            return False
    else:
        print(f"Unsupported OS: {platform.system()}")
        return False


def get_locking_processes(file_path):
    """
    Retrieves a list of processes locking the given file.
    Only implemented for Linux.
    
    Args:
        file_path (str): Path to the locked file.
    
    Returns:
        list: List of process IDs (PIDs) locking the file.
    """
    if platform.system() != "Linux":
        print("Locking process retrieval is only supported on Linux.")
        return []

    try:
        result = subprocess.run(
            ['lsof', '-t', file_path],  # `-t` outputs only PIDs
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip().split()  # List of PIDs
    except FileNotFoundError:
        print("Error: `lsof` command not found. Install it on your system.")
        return []
    

def delete_folder(path):
    """
    Deletes a folder efficiently, handling locked files if deletion fails.
    Includes safeguards to prevent deletion of critical paths.
    
    Args:
        path (str): Path to the folder to delete.
    
    Returns:
        None
    """
    # Safeguard: Prevent deletion of critical paths
    critical_paths = ['/', '/root', '/home', '/var', '/usr', '/bin', '/etc', '.']
    absolute_path = os.path.abspath(path)
    
    if absolute_path in critical_paths or absolute_path.startswith(tuple(critical_paths)):
        print(f"Error: Deletion of critical path '{absolute_path}' is not allowed.")
        return

    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        return

    try:
        # Attempt conventional delete first
        shutil.rmtree(path)
        print(f"Folder '{path}' deleted successfully.")
    except Exception as e:
        print(f"Initial deletion failed for '{path}'. Handling locked files...")
        # Handle locked files individually
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                if is_file_locked(file_path):
                    print(f"File '{file_path}' is locked. Retrieving locking processes...")
                    locking_processes = get_locking_processes(file_path)
                    for pid in locking_processes:
                        try:
                            os.kill(int(pid), 9)  # Forcefully terminate the process
                            print(f"Terminated process {pid} locking '{file_path}'.")
                        except Exception as ex:
                            print(f"Failed to terminate process {pid}: {ex}")
                # Retry deletion
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as ex:
                    print(f"Failed to delete file '{file_path}': {ex}")
            for name in dirs:
                dir_path = os.path.join(root, name)
                try:
                    os.rmdir(dir_path)
                    print(f"Deleted directory: {dir_path}")
                except Exception as ex:
                    print(f"Failed to delete directory '{dir_path}': {ex}")
        # Attempt to delete the root folder again
        try:
            os.rmdir(path)
            print(f"Deleted root folder: {path}")
        except Exception as ex:
            print(f"Failed to delete root folder '{path}': {ex}")


def delete_path(path):
    """
    Safely deletes a file or a folder (recursively if it's a folder).
    
    Args:
        path (str): Path to the file or folder to delete.

    Returns:
        None
    """
    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        return
    
    # Safeguard: Prevent accidental deletion of critical paths
    critical_paths = ['/', 'C:\\', 'C:/', '\\', '.']
    if os.path.abspath(path) in critical_paths:
        print(f"Error: Attempt to delete critical path '{path}' is not allowed.")
        return
    
    try:
        if os.path.isfile(path):
            os.remove(path)
            print(f"File '{path}' has been deleted.")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Folder '{path}' and all its contents have been deleted.")
        else:
            print(f"Error: Path '{path}' is neither a file nor a folder.")
    except Exception as e:
        print(f"Error while deleting '{path}': {e}")
        
        
def extract_tar_file(tar_path, target_folder):
    """
    Extracts a .tar file to the specified target folder.
    If the target folder does not exist or the folder to be created already exists, skips extraction.

    Args:
        tar_path (str): Path to the .tar file.
        target_folder (str): Directory to extract the files into.

    Returns:
        None
    """
    # Check if the .tar file exists
    if not os.path.exists(tar_path):
        print(f"Error: File '{tar_path}' does not exist.")
        return
    
    # Check if the target folder exists
    if not os.path.exists(target_folder):
        print(f"Error: Target folder '{target_folder}' does not exist. Skipping extraction.")
        return

    # Get the folder name that will be created from the .tar file name
    folder_name = os.path.splitext(os.path.basename(tar_path))[0]
    extracted_folder_path = os.path.join(target_folder, folder_name)

    # Check if the folder to be created already exists
    if os.path.exists(extracted_folder_path):
        print(f"Folder '{extracted_folder_path}' already exists. Skipping extraction.")
        return

    # Extract the .tar file
    try:
        print(f"Extracting '{tar_path}' to '{target_folder}'...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=target_folder)
        print(f"Extracted '{tar_path}' successfully to '{target_folder}'.")
    except Exception as e:
        print(f"Error during extraction: {e}")
        

# ------------------- image processing (in place) -------------------

def read_narray_image(image_path: str) -> np.array:
    """
    Read an image from the specified path and return it as a NumPy array.
    """
    with Image.open(image_path) as img:
        return np.array(img)


def rgb_to_grayscale(narray_img: np.array) -> np.array:
    """
    Convert an image in numpy array format from RGB or RGBA to grayscale by averaging all the colors, or return
    the image if it is already in grayscale.
    
    Parameters:
    - narray_img (np.array): Input image in numpy array format.

    Returns:
    - np.array: Grayscale image in numpy array format.
    """
    # Check if the image already is a 2D grayscale image
    if narray_img.ndim == 2:
        return narray_img

    # Check if the input array is a 3D image array
    elif narray_img.ndim == 3:
        if narray_img.shape[-1] == 4:  # If the image has an alpha channel.
            narray_img = narray_img[:, :, :3]
        # Calculate the mean across the color channels
        grayscale_img = np.mean(narray_img, axis=2)
        return grayscale_img
    
    else:
        raise ValueError("Input array must be either a 2D grayscale or a 3D color image array")


def crop_images(image: np.array, regions: list[tuple]) -> list[np.array]:
    """
    Crop multiple regions from an image.

    Args:
    image (np.array): The input image as a NumPy array.
    regions (list of tuples): Each tuple contains two tuples, 
                              defining the top-left and bottom-right 
                              corners of the rectangle to crop (e.g., ((0,0), (66,66))).

    Returns:
    list: A list of np.array, each being a cropped region from the input image.
    """
    cropped_images = []
    for region in regions:
        top_left, bottom_right = region
        # Crop the image using array slicing
        cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        cropped_images.append(cropped_image)
    
    return cropped_images


def split_image(narray_img : np.array, select='') -> Tuple[np.array, np.array]:
    """
    input: image in numpy array format
    output: two images, split in the middle horizontally
    """
    left, right = np.array_split(narray_img, 2, axis=1)
    if select not in ['left', 'right']:
        return left, right
    return left if select == 'left' else right


def join_images(image_list: List[np.array], method='largest') -> np.array:
    """
    Join images side by side with resizing based on the specified method.

    Args:
    image_list (list of np.array): List of images to join.
    method (str): 'largest' to resize all images to the height of the tallest image,
                  'smallest' to resize to the height of the shortest image.

    Returns:
    np.array: A new image consisting of the input images joined side by side.
    """
    if not image_list:
        raise ValueError("image_list cannot be empty")

    # Determine the target height based on the method
    if method == 'largest':
        target_height = max(img.shape[0] for img in image_list)
    elif method == 'smallest':
        target_height = min(img.shape[0] for img in image_list)
    else:
        raise ValueError("Method must be 'largest' or 'smallest'")

    # Resize images and collect them in a list
    resized_images = []
    for img in image_list:
        # Calculate the new width to maintain the aspect ratio
        scale_ratio = target_height / img.shape[0]
        new_width = int(img.shape[1] * scale_ratio)
        resized_img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
        resized_images.append(resized_img)

    # Concatenate all images side by side
    final_image = np.hstack(resized_images)
    return final_image


def subtract_minimum(arr: np.array) -> np.array:
    """
    Subtract the minimum value from each element in a 1D NumPy array.
    Parameters:
    arr (np.ndarray)
    Returns:
    np.ndarray: The processed array with the minimum value subtracted from each element.
    """
    min_value = np.min(arr)
    processed_arr = arr - min_value
    return processed_arr


def minmax_normalization(arr: np.array) -> np.array:
    """
    Min-max normalization
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def image_normalize(narray_img: np.array) -> np.array:
    """
    Normalize the input image by scaling its pixel values to the range [0, 1].
    Parameters:
    image (np.ndarray): A NumPy array representing the input image.
    Returns:
    np.ndarray: The normalized image.
    """
    return narray_img.astype('float32') / 255.


def scale_image(narray_img: np.array, scaling_factor: float=0.5) -> np.array:
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


def resize_image(narray_img: np.array, new_dimensions: Tuple=(256, 256), order=1) -> np.array:
    # order=3: Cubic spline, similar to Lanczos
    return resize(narray_img, new_dimensions, order=order, anti_aliasing=True) 


def resize_image_high_quality(narray_img: np.array, new_dimensions: Tuple=(256, 256)) -> np.array:
    # Convert NumPy array to PIL Image
    image = Image.fromarray(narray_img)
    resized_image = image.resize(new_dimensions, Image.LANCZOS)
    # Convert PIL Image back to NumPy array
    return np.array(resized_image)


# ------------------- arithmatic operations -------------------
def ceil_int_div(a: int, b: int) -> int:
    return -(-a // b)



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

    
def check_and_create_folder(path):
    # Check if the specified path exists
    if not os.path.exists(path):
        # Create the folder if it does not exist
        os.makedirs(path)
        print(f"Folder created at: {path}")
    else:
        print(f"Folder already exists at: {path}")
        

def check_existence(path, if_stop=True, if_create=False):
    if os.path.exists(path):
        print(f"The file or folder '{path}' already exists.")
        print("Exiting the program.")
        if if_stop: sys.exit()  # This will stop the program
    else:
        if if_create:
            check_and_create_folder(path)
        else:
            print(f"The file or folder '{path}' does not exist. Continuing the program.")
            # Continue with the rest of code




