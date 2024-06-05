from .utils import *
from abc import ABC, abstractmethod
from ALP4 import *

import time

# https://github.com/wavefrontshaping/ALP4lib

class DMD(ABC):
    """
    Abstract base class for Digital Micromirror Devices (DMDs).
    Defines the basic interface for getting DMD dimensions.
    """
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with dimensions: {self.get_height()}x{self.get_width()} pixels."
    
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
    
    @abstractmethod
    def display_image(self, image: np.ndarray, bitDepth: int=8) -> None:
        """
        Display an image on the DMD.
        Args:
            image (np.ndarray): The image to display.
            bitDepth (int): The bit depth of the image.
        """
        pass
    
    @abstractmethod
    def end(self) -> None:
        """
        Stop the sequence display and deallocate the device.
        """
        pass
    
    def pad_image(self, img, padding_value=0):
        # Calculate the padding sizes
        pad_height = self.get_height() - img.shape[0]
        pad_width = self.get_width() - img.shape[1]
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
    
    def crop_image(self, img):
        # Calculate the cropping sizes
        crop_height = img.shape[0] - self.get_height()
        crop_width = img.shape[1] - self.get_width()
        # Ensure non-negative cropping sizes
        if crop_height < 0 or crop_width < 0:
            raise ValueError("Original dimensions must be larger than the new image dimensions.")
        # Calculate cropping for height and width
        crop_top = crop_height // 2
        crop_bottom = crop_height - crop_top
        crop_left = crop_width // 2
        crop_right = crop_width - crop_left
        # Apply cropping
        cropped_img = img[crop_top:img.shape[0]-crop_bottom, crop_left:img.shape[1]-crop_right]
        return cropped_img
    
    def adjust_image(self, img, padding_value=0):
        if img.shape[0] < self.get_height() or img.shape[1] < self.get_width():
            return self.pad_image(img, padding_value)
        if img.shape[0] > self.get_height() or img.shape[1] > self.get_width():
            return self.crop_image(img)
        else:
            return img


class ViALUXDMD(DMD):
    def __init__(self, dmd: ALP4=ALP4(version = '4.3')) -> None:
        # Initialize the device
        self.dmd = dmd
        self.dmd.Initialize()
    
    def get_height(self) -> int:
        return self.dmd.nSizeY

    def get_width(self) -> int:
        return self.dmd.nSizeX
    
    def display_image(self, image: np.ndarray, bitDepth: int=8) -> None:
        image = self.adjust_image(image)
        imgSeq = image.ravel()
        # Allocate the onboard memory for the image sequence
        self.dmd.SeqAlloc(nbImg = 1, bitDepth = bitDepth)
        # Send the image sequence as a 1D list/array/numpy array
        self.dmd.SeqPut(imgData = imgSeq)
        # Set image rate to 50 Hz
        self.dmd.SetTiming(pictureTime = 20000) # in microseconds. 50 Hz = 20000 us
        # Run the sequence in a loop
        self.dmd.Run()
        time.sleep(0.01)

    
    def end(self) -> None:
        """
        Stop the sequence display and deallocate the device.
        """
        self.dmd.Halt() 
        self.dmd.FreeSeq()  # Free the sequence from the onboard memory
        self.dmd.Free()  # De-allocate the device
        print("DMD Device deallocated, sequence stopped.")













