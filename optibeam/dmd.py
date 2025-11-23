import time
import cv2
from abc import ABC, abstractmethod
from ALP4 import *

from .utils import timeout, print_underscore


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
        self.pictureTime = 40000  #  in microseconds. 50 Hz = 20000 us, 20ms pictureTime doesn't give the PWM engine enough time to smoothly display 8-bit grayscale!, use 40000 for 25Hz
        # also set the illumination time explicitly
        self.illuminationTime = 38000  # in microseconds
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
    
    @timeout(2)
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
        self.dmd.SetTiming(pictureTime = self.pictureTime, illuminationTime = self.illuminationTime) # in microseconds. 50 Hz = 20000 us
        # Run the sequence in a loop
        self.dmd.Run()
        # time.sleep(0.01)
    
    @timeout(2)
    def get_metadata(self) -> dict:
        config = {}
        config["bit_depth"] = self.bitDepth
        config["picture_time"] = self.pictureTime  
        return config
    
    @timeout(2)
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


def dmd_img_adjustment(display, DMD_DIM, angle=47, horizontal_flip=None, vertical_flip=None):
    # Because the DMD is rotated by about 45 degrees, we need to rotate the generated image by ~45 degrees back
    scale = 1 / np.sqrt(2)
    center = (DMD_DIM // 2, DMD_DIM // 2)
    if horizontal_flip:
        display = cv2.flip(display, 1)
    if vertical_flip:
        display = cv2.flip(display, 0)
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 47 is the angle to rotate to the right orientation in this case
    display = cv2.warpAffine(display, M, (DMD_DIM, DMD_DIM), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0))
    
    return display






import numpy as np
from ALP4 import ALP4

timeout_duration = 3  # seconds
class ViALUXDMD_V2(DMD):
    """
    Drop-in replacement for ViALUXDMD with safer sequence handling.

    API compatibility:
      - __init__(self, dmd: ALP4 | None = None, ...)
      - set_pictureTime, set_bitDepth
      - display_image(image)
      - free_memory()
      - get_metadata()
      - end()
    """

    def __init__(
        self,
        dmd: ALP4 | None = None,
        bitDepth: int = 8,
        pictureTime: int = 40000,      # us
        illuminationTime: int | None = 38000,  # us
    ) -> None:
        super().__init__()

        # Use passed ALP4 handle (your current pattern), or create a new one
        if dmd is None:
            self.dmd = ALP4(version="4.3")
        else:
            self.dmd = dmd

        self.dmd.Initialize()

        self.bitDepth = int(bitDepth)
        self.pictureTime = int(pictureTime)

        if illuminationTime is None:
            self.illuminationTime = self.pictureTime
        else:
            self.illuminationTime = int(illuminationTime)

        # Illumination time must not exceed picture time
        if self.illuminationTime > self.pictureTime:
            self.illuminationTime = self.pictureTime

        self.hight = self.dmd.nSizeY
        self.width = self.dmd.nSizeX

        self._seq_allocated = False
        self._nbImg = 0
        self._closed = False

    # -------- basic getters / setters --------

    def set_pictureTime(self, pictureTime: int) -> None:
        self.pictureTime = int(pictureTime)
        if self.illuminationTime > self.pictureTime:
            self.illuminationTime = self.pictureTime

    def set_bitDepth(self, bitDepth: int) -> None:
        if self._seq_allocated:
            raise RuntimeError(
                "Cannot change bit depth while a sequence is allocated. "
                "Call free_memory() first."
            )
        self.bitDepth = int(bitDepth)

    def get_height(self) -> int:
        return self.hight

    def get_width(self) -> int:
        return self.width

    # -------- internal helpers --------

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust to DMD size and enforce uint8 grayscale, then flatten.
        Keeps your intensity scale (no rescaling, just clip 0–255).
        """
        img = self.adjust_image(image)

        if img.ndim == 3:
            img = img[..., 0]

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img.ravel()

    def _ensure_sequence(self, nbImg: int = 1) -> None:
        """
        Ensure a sequence with nbImg frames exists, reallocating cleanly if needed.
        """
        if self._closed:
            raise RuntimeError("DMD already freed; create a new ViALUXDMD_V2 instance.")

        nbImg = int(nbImg)

        if self._seq_allocated and self._nbImg == nbImg:
            return

        if self._seq_allocated:
            try:
                self.dmd.Halt()
            except Exception:
                pass
            try:
                self.dmd.FreeSeq()
            except Exception:
                pass

        self.dmd.SeqAlloc(nbImg=nbImg, bitDepth=self.bitDepth)
        self._seq_allocated = True
        self._nbImg = nbImg

    # -------- public API --------
    @timeout(timeout_duration)
    def display_image(self, image: np.ndarray) -> None:
        """
        Display a single static image on the DMD.

        Pattern:
          - reuse a 1-frame sequence,
          - Halt() before updating,
          - SeqPut(),
          - SetTiming(),
          - Run().
        """
        imgSeq = self._prepare_image(image)

        self._ensure_sequence(nbImg=1)

        try:
            self.dmd.Halt()
        except Exception:
            pass

        self.dmd.SeqPut(imgData=imgSeq)

        self.dmd.SetTiming(
            pictureTime=self.pictureTime,
            illuminationTime=self.illuminationTime,
        )

        self.dmd.Run()

    @timeout(timeout_duration)
    def free_memory(self) -> None:
        """
        Stop any running sequence and free onboard sequence memory.
        Safe to call after each acquisition as in your main loop.
        """
        if not self._seq_allocated:
            return

        try:
            self.dmd.Halt()
        except Exception:
            pass

        try:
            self.dmd.FreeSeq()
        except Exception:
            pass

        self._seq_allocated = False
        self._nbImg = 0

    @timeout(timeout_duration)
    def get_metadata(self) -> dict:
        return {
            "bit_depth": self.bitDepth,
            "picture_time": self.pictureTime,
            "illumination_time": self.illuminationTime,
            "height": self.hight,
            "width": self.width,
        }

    @timeout(timeout_duration)
    def end(self) -> None:
        """
        Fully release the DMD: Halt, FreeSeq, Free.
        """
        if self._closed:
            return

        if self._seq_allocated:
            try:
                self.dmd.Halt()
            except Exception:
                pass
            try:
                self.dmd.FreeSeq()
            except Exception:
                pass
            self._seq_allocated = False
            self._nbImg = 0

        try:
            self.dmd.Free()
        except Exception:
            pass

        self._closed = True
