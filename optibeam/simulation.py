import matplotlib.pyplot as plt
import numpy as np
import cv2

from typing import *
from abc import ABC, abstractmethod


class DynamicPatterns:
    """
    Class for generating and managing dynamic patterns on a 2D canvas.
    Focusing on image pattern generation only.
    """
    def __init__(self, height: int=64, width: int=64):
        self._height = self._validate_and_convert(height)  # canvas height
        self._width = self._validate_and_convert(width) 
        self.clear_canvas()  # Update canvas size
        self.image = None  # a copy and representation of the canvas
        self._distributions = []
        self.max_pixel_value = 255  # Maximum pixel value for the canvas
        
    def __repr__(self):
        return f"DynamicPatterns with Canvas sides of: {self.canvas.shape}"
    
    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = self._validate_and_convert(value)
        self.clear_canvas()  

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = self._validate_and_convert(value)
        self.clear_canvas()  
        
    def _validate_and_convert(self, value: int) -> int: 
        if not isinstance(value, int):
            try:
                value = int(value)
            except ValueError:
                raise ValueError("Value must be convertible to an integer.")
        if not (0 <= value <= 4096):
            value = 4096 if value > 4096 else 0
            print(f"Value must be between 0 and {4096}.")
        return value
    
    def clear_canvas(self):
        self.canvas = np.zeros((self._height, self._width))

    def apply_distribution(self):
        for dst in self._distributions:
            self.canvas += dst.pattern
            self.canvas = np.clip(self.canvas, 0, self.max_pixel_value)
            
    def update(self):
        """
        Update the canvas by updating all the distributions.
        Need distribution objects have the update method implemented.
        """
        self.clear_canvas()
        for dst in self._distributions:
            dst.update()
        self.apply_distribution()
    
    def append(self, distribution):
        """
        Append a distribution object to the list of distributions.
        """
        self._distributions.append(distribution)
    
    def get_image(self, type="narray"):
        return self.canvas
    
    def transform(self, transformations : List[Callable[[np.ndarray], np.ndarray]]):
        """
        Apply a series of transformations on the final image (), not on the canvas. 
        """
        for transformation in transformations:
            self.canvas = transformation(self.canvas)
    
    def plot_canvas(self, cmap='viridis', pause=0.01):
        plt.clf()
        plt.imshow(self.canvas, cmap=cmap) # cmap='gray' for black and white, and 'viridis' for color
        plt.draw()  
        plt.pause(pause)  # Pause for a short period, allowing the plot to be updated



class Distribution(ABC):
    """
    Abstract class for defining the distribution of different beam patterns.
    """
    def __init__(self, canvas: DynamicPatterns):
        self._height = canvas.height
        self._width = canvas.width
        self._pattern = np.zeros((canvas.height, canvas.width))
        self._transformations = []

    @property
    def pattern(self) -> np.ndarray:
        """Return the distribution's current pattern."""
        return self._pattern

    @abstractmethod
    def update(self):
        """Update the distribution's state."""
        pass



class GaussianDistribution(Distribution):
    """
    Class for generating a 2D Gaussian distribution.
    """
    def __init__(self, canvas: DynamicPatterns, mean_x: float=0.5, mean_y: float=0.5, std_x: float=0.1,
                 std_y: float=0.1, x_velocity: float=0, y_velocity: float=0, speed_momentum: float=0.9,
                 rotation_radians: float=0, rotation_velocity: float=0, rotation_momentum: float=0.95):
        super().__init__(canvas)
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.std_x = std_x
        self.std_y = std_y
        # dynamics, for smooth transition and animation
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.rotation_radians = rotation_radians
        self.rotation_velocity = rotation_velocity
        self.speed_momentum = speed_momentum  # Momentum factor controls the influence of previous changes
        self.rotation_momentum = rotation_momentum

    def change_distribution_params(self, vol_scale: float=0.01, std_scale: float=0.01):
        upper_bound = 1
        lower_bound = 0
        # Calculate new velocity (momentum) for each means
        self.x_velocity = self.speed_momentum * self.x_velocity + np.random.uniform(-vol_scale, vol_scale)
        self.y_velocity = self.speed_momentum * self.y_velocity + np.random.uniform(-vol_scale, vol_scale)
        # Proposed updates for mean positions
        new_mean_x = self.mean_x + self.x_velocity
        new_mean_y = self.mean_y + self.y_velocity
        # Boundary reflection logic
        if new_mean_x < lower_bound or new_mean_x > upper_bound:
            self.x_velocity *= -1  # Reverse and dampen velocity
        if new_mean_y < lower_bound or new_mean_y > upper_bound:
            self.y_velocity *= -1  # Reverse and dampen velocity
        # Update means with possible velocity adjustments
        self.mean_x = np.clip(new_mean_x, lower_bound, upper_bound)
        self.mean_y = np.clip(new_mean_y, lower_bound, upper_bound)
        # Update standard deviations with probabilistic modulation
        change_factor_x = np.random.uniform(-std_scale * self.std_x, std_scale * self.std_x) * (1 - self.std_x / 0.3)
        change_factor_y = np.random.uniform(-std_scale * self.std_y, std_scale * self.std_y) * (1 - self.std_y / 0.3)
        self.std_x = np.clip(self.std_x + change_factor_x, 0.01, 0.3)
        self.std_y = np.clip(self.std_y + change_factor_y, 0.01, 0.3)


    def generate_2d_gaussian(self) -> np.ndarray:
        """
        Generate a rotated 2D Gaussian distribution based on the current state of the distribution.
        """
        # Coordinate grid
        x = np.linspace(0, self._width - 1, self._width)
        y = np.linspace(0, self._height - 1, self._height)
        X, Y = np.meshgrid(x, y)
        # Adjust coordinates relative to center
        X_centered = X - self._width / 2
        Y_centered = Y - self._height / 2
        # Pre-compute cos and sin of rotation angle
        cos_theta, sin_theta = np.cos(self.rotation_radians), np.sin(self.rotation_radians)
        # Apply rotation
        X_rot = cos_theta * X_centered + sin_theta * Y_centered + self._width / 2
        Y_rot = -sin_theta * X_centered + cos_theta * Y_centered + self._height / 2
        
        mean_x = self.mean_x * self._width
        mean_y = self.mean_y * self._height
        std_x = self.std_x * self._width
        std_y = self.std_y * self._height
        return np.exp(-(((X_rot - mean_x) ** 2) / (2 * std_x ** 2) + ((Y_rot - mean_y) ** 2) / (2 * std_y ** 2)))
        
    def update(self):
        self.change_distribution_params()
        self._pattern = self.generate_2d_gaussian()

    
        

class MaxwellBoltzmannDistribution(Distribution):
    """
    Class for generating a 2D Maxwell-Boltzmann distribution.
    """
    pass


class CauchyDistribution(Distribution):
    """
    Class for generating a 2D Cauchy distribution.
    """
    pass
    


# ----------------- 2D narray affine transformation -----------------


# class Transformation:
#     @staticmethod
#     def transformation_matrix_opencv(center, angle, scale, translate):
#         M = cv2.getRotationMatrix2D(center, angle, scale)
#         M[0, 2] += translate[0]
#         M[1, 2] += translate[1]
#         return M

#     def apply_transform(self, image, translate=(0, 0), angle=0, scale=1):
#         rows, cols = image.shape[:2]
#         center = (cols / 2, rows / 2)
#         M = ImageTransformer.transformation_matrix_opencv(center, angle, scale, translate)
#         return cv2.warpAffine(image, M, (cols, rows)) 
    
    
def transform_image(image, rotate=0, scale=1.0, translate=(0, 0), implementation='opencv'):
    """
    Transforms a 2D numpy array image according to the specified parameters.
    - rotate: Rotation angle in degrees (counterclockwise)
    - scale: Scaling factor
    - translate: Tuple (tx, ty) representing translation in pixels
    - implementation: Choice of implementation ('opencv' or 'custom')
    """
    if implementation == 'opencv':
        return _transform_image_opencv(image, rotate, scale, translate)
    else:
        return _transform_image_custom(image, rotate, scale, translate)

def _transform_image_custom(image, rotate, scale, translate):
    maxtrix = compile_transformation_matrix 


def _transform_image_opencv(image, translate=(0, 0), angle=0, scale=1):
    rows, cols = image.shape[:2]
    # Calculate the center for rotation
    center = (cols / 2, rows / 2)
    # Combine rotation and scaling into one matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # Adjust the translation part of the transformation matrix
    M[0, 2] += translate[0]
    M[1, 2] += translate[1]
    # Apply the transformation
    return cv2.warpAffine(image, M, (cols, rows))
    


def compile_transformation_matrix(image: np.ndarray, translate : tuple=(0, 0), radians: float=0,
                                  scale_x: float=1, scale_y: float=1) -> np.ndarray:
    rows, cols = image.shape
    # Center of the image (conceptual origin)
    cx, cy = cols / 2, rows / 2
    # Rotation matrix using radians, centered at image middle
    cos_a = np.cos(radians)
    sin_a = np.sin(radians)
    rotation_matrix = np.array([
        [cos_a, -sin_a, cx * (1 - cos_a) + cy * sin_a],  # Third element recalculated
        [sin_a,  cos_a, cy * (1 - cos_a) - cx * sin_a],  # Third element recalculated
        [0,     0,                                  1]
    ])
    # Scaling matrix with separate x and y scaling, centered at image middle
    scaling_matrix = np.array([
        [scale_x, 0,      cx * (1 - scale_x)],
        [0,      scale_y, cy * (1 - scale_y)],
        [0,      0,                        1]
    ])
    # Translation matrix
    translation_matrix = np.array([
        [1, 0, translate[0]],
        [0, 1, translate[1]],
        [0, 0,            1]
    ])
    # Matrix multiplication order: scale, rotate, then translate
    return translation_matrix @ rotation_matrix @ scaling_matrix


def apply_transformation_matrix(image: np.ndarray, 
                                transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Applies a 3x3 transformation matrix to a 2D numpy array image.
    
    Args:
    - image (numpy.ndarray): The input image as a 2D numpy array.
    - transformation_matrix (numpy.ndarray): The 3x3 transformation matrix.
    
    Returns:
    - numpy.ndarray: The transformed image.
    """
    rows, cols = image.shape[:2]
    return cv2.warpPerspective(image, transformation_matrix, (cols, rows))



# ----------------- image processing functions -----------------

def pixel_value_remap(narray: np.ndarray, max_pixel_value: int=255) -> np.ndarray:
    """
    Rescale the pixel values of the canvas to a new maximum value.
    """
    # Find the minimum and maximum values in the matrix
    min_val, max_val = np.min(narray), np.max(narray)
    # Remap the matrix values to the range [0, 255]
    remapped_matrix = (narray - min_val) / (max_val - min_val) * max_pixel_value
    # Ensure the output is of integer type suitable for image representation
    return remapped_matrix.astype(np.uint8)


def macro_pixel(narray, size: int=8) -> np.ndarray:
    """
    Expand a 2D numpy array (image) to a macro pixel (size, size) array.
    e.g. If canvas is 64x64, and input size is 8, then it will return a 512x512 pixel matrix. 
    
    Parameters:
    - size: The size of the macro pixel.
    
    Returns:
    - A 2D numpy array expanded image.
    """
    # Calculate the new dimensions
    height, width = narray.shape
    new_height = height * size
    new_width = width * size
    # Create a new array for the expanded image
    expanded_image = np.zeros((new_height, new_width))
    for i in range(height):
        for j in range(width):
            expanded_image[i * size : (i+1) * size, 
                           j * size : (j+1) * size] = narray[i, j]
    return expanded_image    


# ----------------- Test functions -----------------

def create_mosaic_image(size: int=1024, n: int=3) -> np.ndarray:
    image = np.zeros((size, size), dtype=float)
    values = np.linspace(0, 255, n**2, dtype=int)
    block_size = size // n
    for i in range(n):
        for j in range(n):
            value_index = i * n + j
            image[i * block_size:(i + 1) * block_size,
                  j * block_size:(j + 1) * block_size] = values[value_index]
    return image


    