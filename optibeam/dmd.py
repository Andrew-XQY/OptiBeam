from .utils import *
from abc import ABC, abstractmethod
from ALP4 import *

# https://github.com/wavefrontshaping/ALP4lib

class DMD(ABC):
    """
    Abstract base class for Digital Micromirror Devices (DMDs).
    Defines the basic interface for getting DMD dimensions.
    """
    
    @abstractmethod
    def get_height(self) -> int:
        """
        Get the height of the DMD in micromirrors or pixels.
        Returns:
            int: The height of the DMD.
        """
        pass

    @abstractmethod
    def get_width(self) -> int:
        """
        Get the width of the DMD in micromirrors or pixels.
        Returns:
            int: The width of the DMD.
        """
        pass


class ViALUXDMD(DMD):
    def __init__(self):
        pass
    
    def get_height(self) -> int:
        return 1024

    def get_width(self) -> int:
        return 1024



def pad_image(img, new_height, new_width, padding_value=0):
    # Calculate the padding sizes
    pad_height = new_height - img.shape[0]
    pad_width = new_width - img.shape[1]
    # Ensure non-negative padding sizes
    if pad_height < 0 or pad_width < 0:
        raise ValueError("New dimensions must be larger than the original image dimensions.")
    # Calculate padding for height and width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    # Apply padding
    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                        mode='constant', constant_values=(padding_value,))
    return padded_img
