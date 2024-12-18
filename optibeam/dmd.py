import time
import cv2
from abc import ABC, abstractmethod
from ALP4 import *


# https://github.com/wavefrontshaping/ALP4lib

class DMD(ABC):
    """
    Abstract base class for Digital Micromirror Devices (DMDs).
    Defines the basic interface for getting DMD dimensions.
    """
    def __init__(self):
        self.name = None
        self.bitDepth = None
        self.pictureTime = None
        self.hight = None
        self.width = None
        
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
    def display_image(self, image: np.ndarray) -> None:
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
    
    @abstractmethod
    def get_metadata(self) -> dict:
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
        self.pictureTime = 20000  #  in microseconds. 50 Hz = 20000 us
        self.bitDepth = 8  # 8-bit grayscale 256 levels
        self.hight = self.dmd.nSizeY
        self.width = self.dmd.nSizeX
        
    def set_pictureTime(self, pictureTime: int) -> None:
        self.pictureTime = pictureTime
    
    def set_bitDepth(self, bitDepth: int) -> None:
        self.bitDepth = bitDepth
    
    def get_height(self) -> int:
        return self.hight

    def get_width(self) -> int:
        return self.width
    
    def free_memory(self) -> None:
        # Stop the sequence display
        self.dmd.Halt()
        # Free the sequence from the onboard memory
        self.dmd.FreeSeq()
    
    def display_image(self, image: np.ndarray) -> None:
        image = self.adjust_image(image)
        imgSeq = image.ravel()
        # Allocate the onboard memory for the image sequence
        self.dmd.SeqAlloc(nbImg = 1, bitDepth = self.bitDepth)
        # Send the image sequence as a 1D list/array/numpy array
        self.dmd.SeqPut(imgData = imgSeq)
        # Set image rate to 50 Hz
        self.dmd.SetTiming(pictureTime = self.pictureTime) # in microseconds. 50 Hz = 20000 us
        # Run the sequence in a loop
        self.dmd.Run()
        # time.sleep(0.01)

    def get_metadata(self) -> dict:
        config = {}
        config["bit_depth"] = self.bitDepth
        config["picture_time"] = self.pictureTime  
        return config
    
    def end(self) -> None:
        """
        Stop the sequence display and deallocate the device.
        """
        try:
            self.dmd.Halt() 
            self.dmd.FreeSeq()  # Free the sequence from the onboard memory
            self.dmd.Free()  # De-allocate the device
            print("DMD Device deallocated, sequence stopped.")
        except:
            pass


def dmd_img_adjustment(display, DMD_DIM, angle=47):
    # Because the DMD is rotated by about 45 degrees, we need to rotate the generated image by ~45 degrees back
    scale = 1 / np.sqrt(2)
    center = (DMD_DIM // 2, DMD_DIM // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 47 is the angle to rotate to the right orientation in this case
    display = cv2.warpAffine(display, M, (DMD_DIM, DMD_DIM), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0))
    return display










