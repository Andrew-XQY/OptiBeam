from .utils import *
from abc import ABC, abstractmethod


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

