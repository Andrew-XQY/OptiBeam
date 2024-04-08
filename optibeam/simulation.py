from .utils import *
from matplotlib.animation import FuncAnimation

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



class GaussianBeam:
    """
    Class used to generate Gaussian beam patterns.
    """
    def __init__(self, mean_x_norm, mean_y_norm, std_x_norm, std_y_norm):
        self.mean_x_norm = mean_x_norm
        self.mean_y_norm = mean_y_norm
        self.std_x_norm = std_x_norm
        self.std_y_norm = std_y_norm
        self.distribution = None

    def generate_2d_gaussian(self, mean_x_norm, mean_y_norm, std_x_norm, std_y_norm):
        # Rescale normalized values to canvas dimensions
        mean_x = mean_x_norm * self._width
        mean_y = mean_y_norm * self._length
        std_x = std_x_norm * self._width
        std_y = std_y_norm * self._length

        x = np.linspace(0, self._width - 1, self._width)
        y = np.linspace(0, self._length - 1, self._length)
        x, y = np.meshgrid(x, y)
        gaussian = np.exp(-(((x - mean_x) ** 2) / (2 * std_x ** 2) + ((y - mean_y) ** 2) / (2 * std_y ** 2)))
        return gaussian


class DynamicPatterns:
    def __init__(self, length=256, width=256):
        self._length = self._validate_and_convert(length)  # number of rows
        self._width = self._validate_and_convert(width)  # number of columns
        self.canvas = np.zeros((self._length, self._width))
        self.items = []
        
    def __repr__(self):
        return f"DynamicPatterns with Canvas sides of: {self.canvas.shape}"
    
    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = self._validate_and_convert(value)
        self.canvas = np.zeros((self._length, self._width))  # Update canvas size

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = self._validate_and_convert(value)
        self.canvas = np.zeros((self._length, self._width))  # Update canvas size
        
    def _validate_and_convert(self, value):
        if not isinstance(value, int):
            try:
                value = int(value)
            except ValueError:
                raise ValueError("Value must be convertible to an integer.")
        if not (0 <= value <= 2048):
            value = 2048 if value > 2048 else 0
            print("Value must be between 0 and 2048.")
        return value

    def apply_distribution(self, gaussian, max_pixel_value=255):
        self.canvas += gaussian
        self.canvas = np.clip(self.canvas, 255, max_pixel_value)

    def continuous_move(self):
        # Placeholder for the continuous movement implementation
        pass
    
    def transform(self, gaussian, transformations):
        # Placeholder for the transformation implementation
        pass
    
    def macro_pixel(self, x, y, size=8):
        """
        Implement of macro pixel on software level
        """
        pass
    
    def save(self, filename):
        pass