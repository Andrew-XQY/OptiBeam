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
    def refresh(self) -> None:
        pass
    
    @abstractmethod
    def grab(self) -> np.ndarray:
        """
        grabs an image from the camera buffer.
        
        Returns:
            np.ndarray: An image array captured from the camera.
        """
        pass


class BaslerCamera(Camera):
    """
    Class representing a Basler camera.
    https://docs.baslerweb.com/pylonapi/pylon-sdk-samples-manual
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
                "IP Address": info.GetIpAddress(),
                "Model": info.GetModelName(), 
                "Serial Number": info.GetSerialNumber(),
                "Device Version": info.GetDeviceVersion(),
                "Device Class": info.GetDeviceClass(),
                "Resolution": (self.camera.Width(), self.camera.Height()),
                "ActionGroupKey": hex(self.camera.ActionGroupKey.Value),
                "ActionGroupMask": hex(self.camera.ActionGroupMask.Value),
                "TriggerSource": self.camera.TriggerSource.Value,
                "TriggerMode": self.camera.TriggerMode.Value,
                "AcquisitionMode": self.camera.AcquisitionMode.Value,
                'Camera grabing status': self.camera.IsGrabbing(),
                'Camera PTP status': self.camera.GevIEEE1588Status.Value
                }
    
    def set_camera_params(self, params: dict):
        """
        Resets the camera parameters based on the input dictionary.

        Parameters:
            params (dict): A dictionary containing camera parameter settings such as exposure, ISO, etc.
        """
        self.camera.ExposureAuto.SetValue(params.get('ExposureAuto') or 'Off')
        self.camera.ExposureTimeRaw.SetValue(1000)  # Set exposure time in microseconds
        self.camera.GainAuto.SetValue(params.get('GainAuto') or 'Off')
        self.camera.GainRaw.SetValue(100)           
        self.camera.GammaEnable.SetValue(params.get('GammaEnable') or True) # (if supported)
        self.camera.Gamma.SetValue(1.0)              
        print(f"Resetting camera with parameters: {params}")
    
    def grab(self):
        pass
    
    def refresh(self):
        pass
                
    def demo_video(self):
        cv2.namedWindow('Camera Output')
        cv2.createTrackbar('Exposure time (ms)', 'Camera Output', 50, 1000, 
                            lambda x: self.camera.ExposureTimeRaw.SetValue(x*1000))  # miniseconds
        for img in self.capture():
            cv2.imshow('Camera Output', img)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key to exit
                break
        cv2.destroyAllWindows()
        
    def ptp_status(self) -> bool:
        """
        https://docs.baslerweb.com/precision-time-protocol#checking-the-status-of-the-ptp-clock-synchronization
        """
        pass
    


class CameraController:
    """
    Class to handle synchronization and simultaneous image capturing from multiple camera objects using PTP and Scheduled Action Commands.
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
