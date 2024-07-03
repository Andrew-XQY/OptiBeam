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
    def __init__(self, height: int=128, width: int=128) -> None:
        self.canvas = None  # The canvas for the dynamic patterns
        self._height = self._validate_and_convert(height)  # canvas height
        self._width = self._validate_and_convert(width) 
        self.clear_canvas()  # Update canvas size
        self.image = None  # a copy and representation of the canvas
        self._distributions = []
        self.max_pixel_value = 255  # Maximum pixel value for the canvas
        
    def __repr__(self) -> str:
        return f"DynamicPatterns with Canvas sides of: {self.canvas.shape}"
    
    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int) -> None:
        self._height = self._validate_and_convert(value)
        self.clear_canvas()  

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int) -> None:
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
    
    def clear_canvas(self) -> None:
        self.canvas = np.zeros((self._height, self._width))
        
    def thresholding(self, threshold: int=5) -> None:
        """
        Apply a threshold to the canvas. Any pixel value below the threshold will be set to 0
        otherwize keep original value.
        """
        # Cut-off pixel value threshold for the canvas (0-255)
        self.canvas[self.canvas < threshold] = 0
        
    def is_blank(self) -> bool:
        """
        Check if the canvas is blank (all zeros).
        """
        return np.all(self.canvas == 0)

    def apply_distribution(self) -> None:
        """
        Apply the pattern of each distribution to the canvas.
        """
        for dst in self._distributions:
            self.canvas += dst.pattern
            self.canvas = np.clip(self.canvas, 0, self.max_pixel_value)
            
    def apply_specific_distribution(self, index: int) -> None:
        """
        Apply the pattern of a specific distribution to the canvas.
        
        args:
        - index: The index of the distribution to be applied.
        
        return: None
        """
        if 0 <= index < len(self._distributions):
            self.canvas += self._distributions[index].pattern
            self.canvas = np.clip(self.canvas, 0, self.max_pixel_value)
            
    def remove_distribution(self, index: int) -> None:
        """
        Remove a distribution object from the list of distributions.
        
        args:
        - index: The index of the distribution to be removed.
        
        return: None
        """
        if 0 <= index < len(self._distributions):
            self._distributions.pop(index)
            
    def remove_distributions(self, amount: int) -> None:
        """
        Remove a number of distributions from the list of distributions.
        
        args:
        - amount: The number of distributions to be removed.
        
        return: None
        """
        if amount >= len(self._distributions):
            self._distributions = []
        else:
            for _ in range(amount):
                self._distributions.pop()
            
    def update(self, *args, **kwargs) -> None:
        """
        Update the canvas by updating all the distributions.
        Need distribution objects have the update method implemented.
        
        args:
        - *args: Variable length argument list.
        - **kwargs: Arbitrary keyword arguments.
        
        return: None
        """
        self.clear_canvas()
        for dst in self._distributions:
            dst.update(*args, **kwargs)
        self.apply_distribution()
        
    def fast_update(self, *args, **kwargs) -> None:
        """
        Call the fast_update method of all the distributions.
        Only update the parameters of the distributions without plotting the new pattern.
        
        args:
        - *args: Variable length argument list.
        - **kwargs: Arbitrary keyword arguments.
        
        return: None
        """
        self.clear_canvas()
        for dst in self._distributions:
            dst.fast_update(*args, **kwargs)
    
    def append(self, distribution) -> None:
        """
        Append a distribution object to the list of distributions.
        
        args:
        - distribution: The distribution object to be appended.
        
        return: None
        """
        self._distributions.append(distribution)
    
    def get_image(self) -> np.ndarray:
        """
        Return a copy of the current canvas
        
        args: None
        
        return: np.ndarray
        """
        return self.canvas
    
    def transform(self, transformations : List[Callable[[np.ndarray], np.ndarray]]) -> None:
        """
        Apply a series of transformations on the final image (), not on the canvas. 
        """
        for transformation in transformations:
            self.canvas = transformation(self.canvas)
    
    def plot_canvas(self, cmap='viridis', pause=0.01) -> None:
        """
        plot the current canvas.
        
        args:
        - cmap: The color map to use for plotting.
        - pause: The pause time for the plot.
        
        return: None
        """
        max_pixel_value = np.max(self.canvas)
        plt.clf()
        plt.imshow(self.canvas, cmap=cmap, vmin=0, vmax=255) # cmap='gray' for black and white, and 'viridis' for color
        plt.colorbar(label='Pixel value')
        plt.title(f'Max Pixel Value: {max_pixel_value}')
        plt.draw()  
        plt.pause(pause)  # Pause for a short period, allowing the plot to be updated
    
    def canvas_pixel_values(self, cmap='gray') -> None:
        fig, ax = plt.subplots()
        im = ax.imshow(self.canvas, cmap=cmap, vmin=0, vmax=255)
        # Function to be called when the mouse is moved
        def on_move(event):
            if event.inaxes == ax:
                x, y = int(event.xdata), int(event.ydata)
                # Get the pixel value of the image at the given (x, y) location
                pixel_value = self.canvas[y, x]
                # Update the figure title with pixel coordinates and value
                ax.set_title(f'Pixel ({x}, {y}): {pixel_value}')
                fig.canvas.draw_idle()
                
        fig.canvas.mpl_connect('motion_notify_event', on_move)
        plt.colorbar(im, ax=ax)  # Shows the color scale
        plt.show()

    def get_metadata(self) -> dict:
        """
        Return the configuration metadata of the dynamic patterns.
        
        args: None
        
        return: dict
        """
        config = {}
        config["simulation_resolution"] = (self._height, self._width)
        config["num_of_distributions"] = len(self._distributions)
        config["types"] = list(set([dst._type for dst in self._distributions]))
        return config


class Distribution(ABC):
    """
    Abstract class for defining the distribution of different beam patterns.
    """
    def __init__(self, canvas: DynamicPatterns):
        self._type = None
        self._height = canvas.height
        self._width = canvas.width
        self._pattern = np.zeros((canvas.height, canvas.width))
        self._transformations = []

    @property
    def pattern(self) -> np.ndarray:
        """Return the distribution's current pattern."""
        return self._pattern

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update the distribution's state."""
        pass
    
    @abstractmethod
    def fast_update(self, *args, **kwargs):
        """Update the distribution's parameters without actually plotting the new pattern."""
        pass
    
    @abstractmethod
    def pattern_generation(self) -> np.ndarray:
        """Generate/plot the distribution's pattern. (in narray form)"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> dict:
        pass



class GaussianDistribution(Distribution):
    """
    Class for generating a 2D Gaussian distribution.
    """
    def __init__(self, canvas: DynamicPatterns, mean_x: float=0.5, mean_y: float=0.5, std_x: float=0.1,
                 std_y: float=0.1, x_velocity: float=0, y_velocity: float=0, speed_momentum: float=0.9,
                 rotation_radians: float=0, rotation_velocity: float=0, rotation_momentum: float=0.95):
        super().__init__(canvas)
        self._type = "Gaussian"
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

    def change_distribution_params(self, vol_scale: float=0.01, std_scale: float=0.01, rot_scale: float=0.01):
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
        # Update rotation angle with momentum
        rotational_adjustment = np.random.uniform(-np.pi*rot_scale, np.pi*rot_scale)  # in radians
        self.rotation_velocity = self.rotation_velocity * self.rotation_momentum + rotational_adjustment
        self.rotation_radians = (self.rotation_radians + self.rotation_velocity) % (2 * np.pi)
        
    def pattern_generation(self) -> np.ndarray:
        """
        Generate a rotated 2D Gaussian distribution based on the current state of the distribution.
        The rotation is centered around the mean of the distribution.
        """
        # Coordinate grid
        x = np.linspace(0, self._width - 1, self._width)
        y = np.linspace(0, self._height - 1, self._height)
        X, Y = np.meshgrid(x, y)
        # Mean coordinates scaled to grid
        mean_x = self.mean_x * self._width
        mean_y = self.mean_y * self._height
        # Adjust coordinates relative to distribution center (mean)
        X_centered = X - mean_x
        Y_centered = Y - mean_y
        # Pre-compute cos and sin of rotation angle
        cos_theta = np.cos(self.rotation_radians)
        sin_theta = np.sin(self.rotation_radians)
        # Apply rotation around the distribution center
        X_rot = cos_theta * X_centered - sin_theta * Y_centered + mean_x
        Y_rot = sin_theta * X_centered + cos_theta * Y_centered + mean_y
        # Compute Gaussian distribution
        std_x = self.std_x * self._width
        std_y = self.std_y * self._height
        return np.exp(-(((X_rot - mean_x) ** 2) / (2 * std_x ** 2) + ((Y_rot - mean_y) ** 2) / (2 * std_y ** 2)))
        
    def update(self, *args, **kwargs):
        self.change_distribution_params(*args, **kwargs)
        self._pattern = self.pattern_generation()
        
    def fast_update(self, *args, **kwargs):
        self.change_distribution_params(*args, **kwargs)

    def get_metadata(self) -> dict:
        return {}
        

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


class StaticGaussianDistribution(Distribution):
    def __init__(self, canvas: DynamicPatterns) -> None:
        super().__init__(canvas)
        self._type = "Static_Gaussian"
        # Gaussian parameters (position, intensity, size and translation)
        self.std_x = 0
        self.std_y = 0
        self.intensity = 0
        self.rotation = 0
        self.dx = 0  # translation in x
        self.dy = 0  # translation in y
        
    def update_params(self, min_std: float=0.03, max_std: float=0.1, max_intensity: int=10, fade_rate: float=0.5) -> None:
        """
        Update the parameters of the Gaussian distribution.
        """
        min_std *= min(self._width, self._height) 
        max_std *= max(self._width, self._height)
        # Random sigmas
        self.std_x = np.random.uniform(low=min_std, high=max_std)
        self.std_y = np.random.uniform(low=self.std_x*0.5, high=self.std_x*1.5)
        # Random intensity with condition (uniform distribution)
        min_intensity = fade_rate * max_intensity/(fade_rate - 1)
        self.intensity = np.random.uniform(min_intensity, max_intensity)
        if self.intensity > 0:  # intensity inversely proportional to area through probabilistic modeling
            area_scaling = (self._width * self._height) / (self.std_x * self.std_y)
            self.intensity += np.random.uniform(0, area_scaling/3)
        # Random Rotation
        angle_degrees = np.random.uniform(0, 360)
        self.rotation = np.deg2rad(angle_degrees)  # Convert angle to radians for rotation
        self.dx = np.random.uniform(0, self._width//2.5)
        self.dy = np.random.uniform(0, self._height//2.5)
        
    def pattern_generation(self) -> np.ndarray:
        """
        Generate a 2D Gaussian distribution based on the current state of the distribution.
        """
        # Check if intensity is negative
        if self.intensity < 0:
            return np.zeros((self._height, self._width))  # Set Gaussian to zero (black canvas)
        else:
        # Create a coordinate grid
            x = np.linspace(0, self._width-1, self._width)
            y = np.linspace(0, self._height-1, self._height)
            x, y = np.meshgrid(x, y)
            # Shift the center for rotation
            x -= self._width/2
            y -= self._height/2
            # Apply rotation
            x_new = x * np.cos(self.rotation) - y * np.sin(self.rotation)
            y_new = x * np.sin(self.rotation) + y * np.cos(self.rotation)
            # Apply translation
            x_new += self._width/2 + self.dx
            y_new += self._height/2 + self.dy
            # Calculate the 2D Gaussian with different sigma for x and y
            dist = np.exp(-((x_new - self._width/2)**2 / (2 * self.std_x**2) + (y_new - self._height/2)**2 / (2 * self.std_y**2)))
            dist *= self.intensity  # Scale Gaussian by the intensity factor
            return dist
        
    def update(self, *args, **kwargs) -> None:
        """Update the distribution's state."""
        self.update_params(*args, **kwargs)
        self._pattern = self.pattern_generation()
    
    def fast_update(self, *args, **kwargs) -> None:
        """Update the distribution's parameters without actually plotting the new pattern."""
        self.update_params(*args, **kwargs)
    
    def get_metadata(self) -> dict:
        return {}

    def demo(self) -> None:
        # Plot the transformed Gaussian
        plt.clf()
        plt.imshow(self._pattern, cmap='viridis')
        plt.colorbar()
        plt.title('Randomly Transformed Anisotropic Gaussian Distribution')
        plt.draw()
        plt.pause(0.5)


class Polygon:
    def __init__(self) -> None:
        pass


class Lens(ABC):
    def __init__(self, canvas: DynamicPatterns, focal_length: float=1.0, aperture: float=1.0):
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
        [cos_a, -sin_a, cx * (1 - cos_a) + cy * sin_a],  
        [sin_a,  cos_a, cy * (1 - cos_a) - cx * sin_a],  
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


def macro_pixel(narray: np.ndarray, size: int=8) -> np.ndarray:
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

def dmd_calibration_pattern_generation(size: int=256, point_size: int=5, boundary_width: int=5) -> np.ndarray:
    # Create a square image with zeros
    image = np.zeros((size, size), dtype=np.uint8)
    # Define the center point
    center = size // 2
    half_point_size = point_size // 2
    image[center-half_point_size:center+half_point_size+1, center-half_point_size:center+half_point_size+1] = 255
    # Draw boundaries
    image[:boundary_width, :] = 255  # Top boundary
    image[-boundary_width:, :] = 255  # Bottom boundary
    image[:, :boundary_width] = 255  # Left boundary
    image[:, -boundary_width:] = 255  # Right boundary
    return image

def dmd_calibration_gradient(size: int=128, point_size: int=5, boundary_width: int=5) -> np.ndarray:
    # Create a square image with gradient background
    image = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    # Define the center point
    center = size // 2
    half_point_size = point_size // 2
    image[center-half_point_size:center+half_point_size+1, center-half_point_size:center+half_point_size+1] = 255
    # Draw boundaries
    image[:boundary_width, :] = 255  # Top boundary
    image[-boundary_width:, :] = 255  # Bottom boundary
    image[:, :boundary_width] = 255  # Left boundary
    image[:, -boundary_width:] = 255  # Right boundary
    return image

def dmd_calibration_corner_dots(size=256, dot_size=10):
    # Create a blank canvas
    image = np.zeros((size, size), dtype=np.uint8)
    # Positions for the center of the dots in each corner
    corners = [
        (dot_size, dot_size),  # Top-left corner
        (size - dot_size - 1, dot_size),  # Top-right corner
        (dot_size, size - dot_size - 1),  # Bottom-left corner
        (size - dot_size - 1, size - dot_size - 1)  # Bottom-right corner
    ]
    # Draw dots by setting pixels within the dot radius to maximum intensity
    for x, y in corners:
        for i in range(-dot_size, dot_size + 1):
            for j in range(-dot_size, dot_size + 1):
                if i**2 + j**2 <= dot_size**2:
                    image[y + i, x + j] = 255  # Ensure we're within bounds automatically due to numpy handling
    return image

def dmd_calibration_center_dot(size=256, dot_size=10):
    # Create a square canvas filled with zeros
    canvas = np.zeros((size, size), dtype=np.uint8)
    # Calculate the center position
    center = size // 2
    # Calculate the coordinates for the dot
    start = center - dot_size // 2
    end = center + dot_size // 2
    # Draw the dot on the canvas
    canvas[start:end, start:end] = 255
    return canvas

def generate_mosaic_image(size: int=1024, n: int=3) -> np.ndarray:
    image = np.zeros((size, size), dtype=float)
    values = np.linspace(0, 255, n**2, dtype=int)
    block_size = size // n
    for i in range(n):
        for j in range(n):
            value_index = i * n + j
            image[i * block_size:(i + 1) * block_size,
                  j * block_size:(j + 1) * block_size] = values[value_index]
    return image

def generate_radial_gradient(size: int=256):
    # Create an empty array of the specified dimensions
    image = np.zeros((size, size), dtype=np.float32)
    # Calculate the center coordinates
    center_x, center_y = size // 2, size // 2
    # Maximum distance from the center to a corner (radius for decay)
    max_radius = np.sqrt(center_x**2 + center_y**2)
    # Populate the array with intensity values based on radial distance
    for x in range(size):
        for y in range(size):
            # Calculate distance from the current pixel to the center
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            # Normalize the distance and calculate intensity
            if distance <= max_radius:
                intensity = 255 * (1 - distance / max_radius)
                image[x, y] = intensity
    return image.astype(np.uint8)

def generate_upward_arrow(size=256):
    canvas = np.zeros((size, size), dtype=np.uint8)
    center_x = size // 2
    center_y = size // 2
    # Arrow dimensions
    arrow_width = size // 10
    arrow_height = size // 2
    head_height = size // 6
    head_width = size // 6
    # Draw the arrow shaft
    shaft_start_y = center_y + arrow_height // 2
    shaft_end_y = center_y - arrow_height // 2
    canvas[shaft_end_y:shaft_start_y, center_x - arrow_width // 2:center_x + arrow_width // 2] = 255
    # Draw the arrow head
    head_start_y = shaft_end_y - size // 6
    for i in range(head_height):
        start_x = center_x - (head_width // 2) * (i / head_height)
        end_x = center_x + (head_width // 2) * (i / head_height)
        canvas[head_start_y + i:head_start_y + i + 1, int(start_x):int(end_x)] = 255
    return canvas

def generate_solid_circle(size=256):
    image = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 2
    cv2.circle(image, center, radius, 255, -1)
    return image


# ----------------- Image generator functions -----------------
def moving_blocks_generator(size: int=256, block_size: int=32, intensity: int=255):
    num_blocks = size // block_size
    base_image = np.zeros((size, size), dtype=np.uint8)
    while True:
        for i in range(num_blocks):
            for j in range(num_blocks):
                img_array = base_image.copy()
                img_array[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = intensity
                yield img_array
                
def position_intensity_generator(size: int=256, block_size: int=32, 
                                 intensity: int=50, intensity_step: int=10):
    num_blocks = size // block_size
    base_image = np.zeros((size, size), dtype=np.uint8)
    while True:
        for i in range(num_blocks):
            for j in range(num_blocks):
                img_array = base_image.copy()
                img_array[i*block_size:(i+1)*block_size, 
                        j*block_size:(j+1)*block_size] = intensity
                nonzero_mask = img_array > 0
                while np.max(img_array) + intensity_step <= 255:
                    new_image = img_array.copy()
                    new_image[nonzero_mask] = np.clip(img_array[nonzero_mask] + intensity_step, 0, 255).astype(np.uint8)
                    yield new_image
                    img_array = new_image  # Update image to newly adjusted image

