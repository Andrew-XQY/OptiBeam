from abc import ABC, abstractmethod
import numpy as np
import cv2
from pypylon import pylon
import time


class MultiBaslerCameraManager:

    def __init__(self, params={"action_key" : 0x1, "group_key" : 0x1, "group_mask" : 0xffffffff, "boardcast_ip" : "255.255.255.255"}):
        self.cameras = []
        self.flip = False
        self.tlFactory = pylon.TlFactory.GetInstance()
        self.GigE_TL = self.tlFactory.CreateTl('BaslerGigE') # GigE transport layer, used for issuing action commands
        self.action_key = params.get("action_key")
        self.group_key = params.get("group_key")
        self.group_mask = params.get("group_mask")  # pylon.AllGroupMask or 0xffffffff
        self.boardcast_ip = params.get("boardcast_ip") # Broadcast to all devices in the network
        self.initialize()
        
    def _start_grabbing(self):
        for cam in self.cameras:
            cam.StartGrabbing() # pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser
    
    def _stop_grabbing(self):
        for cam in self.cameras:
            cam.StopGrabbing()
            
    def _combine_images(self, im0, im1):
        return np.hstack((im0, im1) if not self.flip else (im1, im0))

    def _flip(self):
        cv2.namedWindow('Acquisition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Acquisition', 1280, 512)
        self._start_grabbing()
        while all(cam.IsGrabbing() for cam in self.cameras):
            grabResult0 = self.cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            grabResult1 = self.cameras[1].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult0.GrabSucceeded() and grabResult1.GrabSucceeded():
                im0 = grabResult0.GetArray()
                im1 = grabResult1.GetArray()
                combined_image = self._combine_images(im0, im1)
                cv2.imshow('Acquisition', combined_image)
                key = cv2.waitKey(1)
                if key == ord('f'):  # 'f' key to flip
                    self.flip = not self.flip
                    print('Image logic flipped.')
                elif key == 27:  # Escape key
                    break
        grabResult0.Release()
        grabResult1.Release()
        self._stop_grabbing()
        cv2.destroyAllWindows()

    def initialize(self, timeout=10):
        """
        detect all cameras and initialize them
        """
        start_time = time.time()
        timeout += start_time
        while True:
            devices = self.tlFactory.EnumerateDevices()
            if len(devices) >= 2:
                break
            if time.time() >= timeout:
                raise RuntimeError(f"At least 2 cameras are required. Detected: {len(devices)}.")
            time.sleep(0.5)
        
        for i in range(len(devices)):  
            camera = pylon.InstantCamera(self.tlFactory.CreateDevice(devices[i]))
            camera.Open()
            camera.AcquisitionFrameRateEnable.Value = True
            camera.AcquisitionFrameRateAbs.Value = 20.0
            self.cameras.append(camera)
        
        self._flip()
        
        for i in self.cameras:  # prepare for PTP and scheduled action command
            i.AcquisitionFrameRateEnable.Value = False
            i.GevIEEE1588.Value = True
            i.AcquisitionMode.SetValue("SingleFrame") # SingleFrame Continuous
            i.TriggerMode.SetValue("On")
            i.TriggerSource.SetValue("Action1")
            i.TriggerSelector.SetValue('FrameStart')
            i.ActionDeviceKey.SetValue(self.action_key)
            i.ActionGroupKey.SetValue(self.group_key)
            i.ActionGroupMask.SetValue(self.group_mask)
            
        self.print_all_camera_status()

    def print_all_camera_status(self):
        print('-' * 50)
        print(f"Number of cameras detected: {len(self.cameras)}")
        print('\n')
        for cam in self.cameras:
            info = cam.GetDeviceInfo()
            print("Using %s @ %s @ %s" % (info.GetModelName(), info.GetSerialNumber(), info.GetIpAddress()))
            print("ActionGroupKey:", hex(cam.ActionGroupKey.Value))
            print("ActionGroupMask:", hex(cam.ActionGroupMask.Value))
            print("TriggerSource:", cam.TriggerSource.Value)
            print("TriggerMode:", cam.TriggerMode.Value)
            print("AcquisitionMode:", cam.AcquisitionMode.Value)
            print('Camera grabing status: ', cam.IsGrabbing())
            print('Camera PTP status: ', cam.GevIEEE1588Status.Value)
            print('\n')
        print('-' * 50)
        
    def _check_cameras_state(self, timeout=10):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if all(camera.GevIEEE1588Status.Value in ['Slave', 'Master'] for camera in self.cameras):
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

    def synchronization(self, threshold=300, timeout=20):
        self._check_cameras_state()
        print('Waiting for PTP time synchronization...')
        offset = float('inf')
        records = []
        start_time = time.time()
        while time.time() - start_time < timeout:
            offset = self.check_sync_status()
            records.append(offset)
            print(offset)
            offset = abs(offset)
            if offset < threshold: 
                print("Cameras synchronized.")
                return records
            time.sleep(1)
        raise TimeoutError("Cameras did not synchronize within the timeout period.")
    
    def max_time_difference(self, timestamps: list):
        return max(timestamps) - min(timestamps)
        
    def schedule_action_command(self, scheduled_time: int = 3000000000):
        self.cameras[0].GevTimestampControlLatch.Execute() # Get the current timestamp from the camera
        current_time = self.cameras[0].GevTimestampValue.Value
        scheduled_time = current_time + scheduled_time  # Define the delay for action command (in nanoseconds)
        self._start_grabbing()
        # Issue the scheduled action command
        results = self.GigE_TL.IssueScheduledActionCommandNoWait(self.action_key, self.group_key, self.group_mask, scheduled_time, self.boardcast_ip)
        print(f"Scheduled command issued, retriving image...")
        grabResult0 = self.cameras[0].RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
        grabResult1 = self.cameras[1].RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
        if grabResult0.GrabSucceeded() & grabResult1.GrabSucceeded():
            im0 = grabResult0.GetArray()
            im1 = grabResult1.GetArray()
            t0 = grabResult0.TimeStamp
            t1 = grabResult1.TimeStamp
            timedif = self.max_time_difference([t0, t1])
            print(f"Camera image captured time difference: {t0 - scheduled_time} ns")
            print(f"Time difference between two images: {timedif} ns \n")
            if timedif < 1000:
                combined_image = self._combine_images(im0, im1)
            else: combined_image = None
        grabResult0.Release()
        grabResult1.Release()
        self._stop_grabbing() 
        return combined_image
    
    def perodically_scheduled_action_command(self, save_path: str, total: int = 10, wait_time: int = 1000):
        for _ in range(total):
            image = self.schedule_action_command(int(wait_time * 1e6))
            if image is not None:
                cv2.imwrite(save_path + str(_) + '.png', image)  
    
    def end(self):
        self._stop_grabbing()
        for cam in self.cameras:
            cam.Close()
        print("Camera closed, grab terminated.")
        
        

# class Camera(ABC):
#     def __init__(self, camera=None):
#         self.role = 'unsigned'  # reserved for PTP synchronization
#         self.camera = camera  # wrapped camera object of the specific camera brand
#         self.parameters = {}

#     @property
#     def role(self):
#         """
#         Get the role of the camera.
#         """
#         return self._role

#     @role.setter
#     def role(self, value):
#         """
#         Set the role of the camera with validation.
#         """
#         if value not in ['beam_image', 'fiber_output', 'unsigned']:
#             raise ValueError("Role must be 'beam_image' or 'fiber_output' or 'unsigned'.")
#         self._role = value
        
#     @abstractmethod
#     def set_parameter(self, **kwargs):
#         """
#         Set a camera parameter.
#         """
#         pass

#     @abstractmethod
#     def get_information(self):
#         """
#         Return a dictionary containing camera information.
#         """
#         pass
    
#     @abstractmethod
#     def demo(self):
#         """
#         freerun image streaming
#         """
#         pass


# class BaslerCamera(Camera):
#     def __init__(self, camera=None):
#         super().__init__(camera)
#         self.grabResult = None
        
#     def demo(self):
#         """
#         Generator method to continuously yield images as they are captured by the camera.
#         """
#         try:
#             self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser)
#             while self.camera.IsGrabbing():
#                 grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#                 if grabResult.GrabSucceeded():
#                     img = grabResult.GetArray()
#                     yield img
#                 grabResult.Release()
#         finally:
#             # Ensure the camera stops grabbing and properly releases resources when the generator stops.
#             self.camera.StopGrabbing()

#     def set_parameter(self, **kwargs):
#         pass

#     def get_information(self):
#         return {}
    
#     def refresh(self):
#         """
#         Refresh the camera (not restart Main purpose is to make sure the camera is ready for the next action command)
#         """
#         pass
