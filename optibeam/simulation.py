from .utils import *
from .dmd import *
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class DynamicPatterns:
    """
    Class for generating and managing dynamic patterns on a 2D canvas.
    """
    def __init__(self, display, height: int=64, width: int=64):
        self._max_height = display.get_height()
        self._max_width = display.get_width()
        self._height = self._validate_and_convert(height)  # canvas height
        self._width = self._validate_and_convert(width) 
        self.clear_canvas()  # Update canvas size
        self._distributions = []
        
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
        MAX = max(self._max_height, self._max_width)
        if not (0 <= value <= MAX):
            value = MAX if value > MAX else 0
            print(f"Value must be between 0 and {MAX}.")
        return value
    
    def clear_canvas(self):
        self.canvas = np.zeros((self._height, self._width))

    def apply_distribution(self, max_pixel_value: int=255):
        for dst in self._distributions:
            self.canvas += dst.pattern
            self.canvas = np.clip(self.canvas, 0, max_pixel_value)
            
    def update(self):
        self.clear_canvas()
        for dst in self._distributions:
            dst.update()
        self.apply_distribution()
    
    def macro_pixel(self, size: int=8):
        """
        Expand a 2D numpy array (image) to a macro pixel (size, size) array.
        e.g. If canvas is 64x64, and input size is 8, then it will return a 512x512 pixel matrix. 
        If size overflows the display dimension, it will automatically rezised to the maximum possible one.
        
        Parameters:
        - size: The size of the macro pixel.
        
        Returns:
        - A 2D numpy array representing the expanded image.
        """
        # Calculate the potential new dimensions
        new_height = self._height * size
        new_width = self._width * size
        # Adjust macro_pixel_size if new dimensions exceed max_dimension
        if new_height > self._max_height or new_width > self._max_width:
            size = max(self._max_height, self._max_width) // max(self._height, self._width)
            new_height = self._height * size
            new_width = self._width * size
        # Create a new array for the expanded image
        expanded_image = np.zeros((new_height, new_width))
        for i in range(self._height):
            for j in range(self._width):
                expanded_image[i * size : (i+1) * size, 
                               j * size : (j+1) * size] = self.canvas[i, j]
        return expanded_image
    
    def append(self, distribution):
        self._distributions.append(distribution)
    
    def export(self, type="narray"):
        pass
    
    def transform(self, transformations : List[Callable[[np.ndarray], np.ndarray]]):
        """
        Apply a series of transformations to the distribution's pattern on the canvas level.
        Do not actually change the distribution's data (_pattern property) but change the final appearance on the canvas.
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
    
    def add_transformation(self, transformation: callable):
        """Add a transformation function to the list."""
        self._transformations.append(transformation)

    def transform(self):
        """Apply transformations (matrix) to the pattern. such as rotation, scaling, etc."""
        for func in self._transformations:
            self._pattern = func(self._pattern)



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
        # Calculate new velocity (momentum) for each means
        self.x_velocity = self.speed_momentum * self.x_velocity + np.random.uniform(-vol_scale, vol_scale)
        self.y_velocity = self.speed_momentum * self.y_velocity + np.random.uniform(-vol_scale, vol_scale)
        # Update parameters
        self.mean_x = np.clip(self.mean_x + self.x_velocity, 0.1, 0.9)
        self.mean_y = np.clip(self.mean_y + self.y_velocity, 0.1, 0.9)
        self.std_x = np.clip(self.std_x + np.random.uniform(-std_scale * self.std_x, std_scale * self.std_x), 0, 0.3)
        self.std_y = np.clip(self.std_y + np.random.uniform(-std_scale * self.std_y, std_scale * self.std_y), 0, 0.3)

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
    
    def apply_transition(self):
        pass
        
    def apply_rotation(self):
        rotational_adjustment = np.random.uniform(-np.pi/180, np.pi/180)  # Between -1 and +1 degree in radians
        self.rotation_velocity = self.rotation_velocity * self.rotation_momentum + rotational_adjustment
        self.rotation_radians = (self.rotation_radians + self.rotation_velocity) % (2 * np.pi)
        
    def update(self):
        self.change_distribution_params()
        self._pattern = self.generate_2d_gaussian()
        self.transform()
        if self.rotation_radians != 0:
            self.apply_rotation()
    
        

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
    
    
class Converter:
    """
    Class used to convert generated beam pattern (image) into different formats including Python objects and files.
    where divices like DMD, SLM can load the beam pattern.
    """
    def __init__(self, name, input_type, output_type, function):
        self.name = name
        self.input_type = input_type
        self.output_type = output_type
        self.function = function

    def convert(self, value):
        return self.function(value)

