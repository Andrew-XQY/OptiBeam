from .utils import *
from abc import ABC, abstractmethod
import numpy as np
import cv2
from pypylon import pylon
from datetime import datetime


class Camera(ABC):
    """
    Abstract base class for a camera, providing a blueprint for camera operations.
    """
    
    @abstractmethod
    def get_info(self) -> dict:
        """
        Retrieves information about the camera.

        Returns:
            dict: A dictionary containing camera details such as model, resolution, and other relevant parameters.
        """
        pass
    
    @abstractmethod
    def set_camera_params(self, params: dict):
        """
        Resets the camera parameters based on the input dictionary.

        Parameters:
            params (dict): A dictionary containing camera parameter settings such as exposure, ISO, etc.
        """
        pass
    
    @abstractmethod
    def ptp_status(self) -> bool:
        """
        Checks the status of the Precision Time Protocol (PTP) on the camera.

        Returns:
            bool: True if PTP is enabled, False otherwise.
        """
        pass
    
    @abstractmethod
    def enable_ptp(self) -> None:
        """
        Enables the Precision Time Protocol (PTP) on the camera. (if supported)
        """
        pass
    
    @abstractmethod
    def open(self) -> None:
        """
        Opens the camera for capturing images.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Closes the camera after capturing images. release resources.
        """
        pass
    
    @abstractmethod
    def capture(self) -> np.ndarray:
        """
        Captures an image using the camera.

        Returns:
            np.ndarray: An array representing the captured image.
        """
        pass


class BaslerCamera(Camera):
    """
    Class representing a Basler camera.
    https://docs.baslerweb.com/precision-time-protocol#checking-the-status-of-the-ptp-clock-synchronization
    """
    
    def __init__(self, camera: pylon.InstantCamera, params: dict={}):
        """
        Initializes a Basler camera object with a given camera ID.

        Parameters:
            camera_id (int): The ID of the camera.
        """
        self.camera = camera
        self.open()
        self.set_camera_params(params)

    def open(self):
        if self.camera is not None:
            try:
                self.camera.Close()
            except Exception:
                pass
        self.camera.Open()
        
    def close(self):
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        self.camera.Close()
    
    def get_info(self) -> dict:
        """
        Retrieves information about the Basler camera.
        
        Returns:
            dict: A dictionary containing camera details such as model, serial number, etc.
        """
        info = self.camera.GetDeviceInfo()
        return {
                "Camera ID": self.camera_id,
                "Camera Information": info.GetModelName(), 
                "Camera Serial Number": info.GetSerialNumber(),
                "Camera Device Version": info.GetDeviceVersion(),
                "Camera Device Class": info.GetDeviceClass(),
                "Camera Resolution": (self.camera.Width(), self.camera.Height())
                }
    
    def set_camera_params(self, params: dict):
        """
        Resets the camera parameters based on the input dictionary.

        Parameters:
            params (dict): A dictionary containing camera parameter settings such as exposure, ISO, etc.
        """
        self.converter = pylon.ImageFormatConverter()
        # Setting the converter to output mono8 images for simplicity
        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.output_dim = [self.camera.Width.GetValue(), self.camera.Height.GetValue()]
        
        # Ensure the camera exposure, gain, and gamma are set to manual mode before adjusting
        self.camera.ExposureAuto.SetValue('Off')  # Turn off auto exposure
        self.camera.GainAuto.SetValue('Off')      # Turn off auto gain
        self.camera.GammaEnable.SetValue(True)    # Enable gamma correction if supported
        
        # Adjust camera settings - these values are examples and should be adjusted based on your needs and camera capabilities
        self.camera.ExposureTimeRaw.SetValue(100000)  # Set exposure time to 40000 microseconds
        self.camera.GainRaw.SetValue(100)            # Set gain
        self.camera.Gamma.SetValue(1.0)              # Set gamma value to 1.0 (if supported)

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print(f"Resetting camera parameters: {params}")
    
    def capture(self) -> Generator[np.ndarray, None, None]:
        while True:  # Change this to a more reliable condition if necessary
            if not self.camera.IsGrabbing():
                self.open_camera()
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            try:
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    # Convert to OpenCV format
                    image = self.converter.Convert(grabResult)
                    img = image.GetArray()
                    yield img
                grabResult.Release()
            except Exception as e:
                print("Error encountered: ", e)
                # Optionally, attempt to reconnect or handle error
                img = cv2.putText(np.zeros((self.output_dim[1], self.output_dim[0]), np.uint8),
                                  "No Image Input", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                yield img
                # No need to break here; let it attempt to reconnect in the next iteration
                
    def demo_video(self):
        save_to = "../../ResultsCenter/images"
        cv2.namedWindow('Camera Output')
        cv2.createTrackbar('Exposure time (ms)', 'Camera Output', 50, 1000, 
                            lambda x: self.camera.ExposureTimeRaw.SetValue(x*1000))  # miniseconds
        for img in self.capture():
            cv2.imshow('Camera Output', img)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key to exit
                break
            elif key == ord('s'):  # 's' key to save the image
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{save_to}/{timestamp}.png"
                cv2.imwrite(filename, img)
                print(f"Image saved as {filename}")
        
    def ptp_status(self) -> bool:
        pass
    
    def enable_ptp(self) -> None:
        pass



class Synchronizer:
    """
    Class to handle synchronization and simultaneous image capturing from multiple camera objects using PTP.
    """
    
    def __init__(self, cameras: List[Camera]):
        """
        Initializes the Synchronizer with a list of Camera objects.

        Parameters:
            cameras (List[Camera]): A list of camera objects to be synchronized and managed.
        """
        self.cameras = cameras
        self.initialize_ptp()
    
    def camera_registration(self):
        pass

    def initialize_ptp(self):
        """
        Initializes PTP on all cameras to synchronize them.
        """
        pass
    
    def take_images(self):
        pass


# ------------------- other functionalities -------------------

def num_of_cameras_detected():
    """
    Returns the detected number of cameras that connected to the computer

    Returns:
        int: The number of cameras detected.
    """
    # Get the transport layer factory
    tl_factory = pylon.TlFactory.GetInstance()
    # Get all attached devices
    devices = tl_factory.EnumerateDevices()
    if len(devices) == 0:
        print("No cameras detected.")
    else:
        print("Number of cameras detected:", len(devices))
        # Print out the device info for each detected camera
        for i, device in enumerate(devices):
            print(f"Camera {i + 1}: {device.GetModelName()} - Serial Number: {device.GetSerialNumber()}")
    return len(devices)
