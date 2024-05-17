from abc import ABC, abstractmethod
import numpy as np
import cv2
from pypylon import pylon
import time

class Camera(ABC):

    def __init__(self):
        self.parameters = {}

    @abstractmethod
    def set_params(self, params):
        pass

    @abstractmethod
    def update_params(self, **params):
        pass

    @abstractmethod
    def is_ready(self):
        pass

    def get_params(self):
        return self.parameters
    

class BaslerCamera(Camera):

    def __init__(self, camera: pylon.InstantCamera):
        super().__init__()
        self.ready = False
        self.camera = camera
        
    def camera_status(self):
        info = self.camera.GetDeviceInfo()
        print('-' * 50)
        print("Using %s @ %s @ %s" % (info.GetModelName(), info.GetSerialNumber(), info.GetIpAddress()))
        print("ActionGroupKey:", hex(self.camera.ActionGroupKey.Value))
        print("ActionGroupMask:", hex(self.camera.ActionGroupMask.Value))
        print("TriggerSource:", self.camera.TriggerSource.Value)
        print("TriggerMode:", self.camera.TriggerMode.Value)
        print("AcquisitionMode:", self.camera.AcquisitionMode.Value)
        print('Camera grabing status: ', self.camera.IsGrabbing())
        print('Camera PTP status: ', self.camera.GevIEEE1588Status.Value)
        print('-' * 50)

    def set_params(self, params):
        for key, value in params.items():
            if key in self.parameters:
                self.parameters[key] = value

    def update_params(self, **params):
        for key, value in params.items():
            self.parameters[key] = value

    def start_camera(self):
        pass
        


class MultiBaslerCameraManager:

    def __init__(self, action_key: int = 0x1, group_key: int = 0x1, group_mask: int = 0xffffffff, boardcast_ip: str = "255.255.255.255"):
        self.cameras = None
        self.tlFactory = pylon.TlFactory.GetInstance()
        self.GigETL = self.tlFactory.CreateTl('BaslerGigE') # GigE transport layer, used for issuing action commands
        self.action_key = action_key
        self.group_key = group_key
        self.group_mask = group_mask  # pylon.AllGroupMask or 0xffffffff
        self.boardcast_ip = boardcast_ip # Broadcast to all devices in the network
        self.initialize()

    def initialize(self, timeout=10):
        """
        detect all cameras and initialize them
        """
        start_time = time.time()
        while start_time < timeout:
            devices = self.tlFactory.EnumerateDevices()
            if len(devices) == 2:
                break
            time.sleep(0.5)
        if start_time >= timeout:
            raise RuntimeError("Not enough cameras detected in the network.")
        self.cameras = pylon.InstantCameraArray(len(devices))
        for i, camera in enumerate(self.cameras):  # prepare for PTP and scheduled action command
            camera.Attach(self.tlFactory.CreateDevice(devices[i]))
            camera.Open()
            camera.GevIEEE1588.Value = True
            camera.AcquisitionMode.SetValue("SingleFrame") # SingleFrame Continuous
            camera.TriggerMode.SetValue("On")
            camera.TriggerSource.SetValue("Action1")
            camera.TriggerSelector.SetValue('FrameStart')
            camera.ActionDeviceKey.SetValue(self.action_key)
            camera.ActionGroupKey.SetValue(self.group_key)
            camera.ActionGroupMask.SetValue(self.group_mask)

    def check_cameras_state(self, timeout=10):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if all(camera.GevIEEE1588Status.Value in ['Slave', 'Master'] for camera in self.cameras):
                print("All cameras are ready.")
                return True
            time.sleep(0.5)
        raise TimeoutError("Cameras did not become ready within the timeout period.")

    def check_sync_status(self):
        # check PTP offset anytime
        slave = None
        for i, camera in enumerate(self.cameras):
            if camera.GevIEEE1588Status.Value == 'Slave':
                slave = i
        self.cameras[0].GevIEEE1588DataSetLatch.Execute()
        self.cameras[1].GevIEEE1588DataSetLatch.Execute()
        return self.cameras[slave].GevIEEE1588OffsetFromMaster.Value

    def synchronization(self, threshold=200, timeout=10):
        self.check_cameras_state()
        print('Waiting for PTP time synchronization...')
        offset = float('inf')
        records = []
        start_time = time.time()
        while time.time() - start_time < timeout:
            offset = self.check_sync_status()
            records.append(offset)
            print(offset)
            offset = abs(offset)
            if offset > threshold: return records
            time.sleep(1)
        raise TimeoutError("Cameras did not synchronize within the timeout period.")
    
    def schedule_action_command(self):
        pass
    
    def perodically_scheduled_action_command(self):
        pass