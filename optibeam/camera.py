from abc import ABC, abstractmethod
import numpy as np
import cv2
from pypylon import pylon
import time


def create_camera_control_functions(camera):
    """
    Callback functions to set exposure and gain using closures
    """
    def set_exposure(val):
        camera.ExposureTimeAbs.Value = val
    def set_gain(val):
        camera.GainRaw.Value = val
    return {'Exposure' : set_exposure, 'Gain' : set_gain}
    
class MultiBaslerCameraManager:

    def __init__(self, params={"action_key" : 0x1, "group_key" : 0x1, 
                               "group_mask" : 0xffffffff, "boardcast_ip" : "255.255.255.255"}):
        self.cameras = []
        self.flip = False
        self.master = None
        self.tlFactory = pylon.TlFactory.GetInstance()
        self.GigE_TL = self.tlFactory.CreateTl('BaslerGigE') # GigE transport layer, used for issuing action commands
        self.action_key = params.get("action_key")
        self.group_key = params.get("group_key")
        self.group_mask = params.get("group_mask")  # pylon.AllGroupMask or 0xffffffff
        self.boardcast_ip = params.get("boardcast_ip") # Broadcast to all devices in the network
        self.initialize()
        
    def _start_grabbing(self) -> None:
        for cam in self.cameras:
            cam.StartGrabbing() # pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser
    
    def _stop_grabbing(self) -> None:
        for cam in self.cameras:
            cam.StopGrabbing()
            
    def _combine_images(self, im0 :np.ndarray, im1 :np.ndarray) -> np.ndarray:
        return np.hstack((im0, im1) if not self.flip else (im1, im0))

    def _camera_params_setting(self, cv2_window_name: str) -> None:
        """
        mount the camera control functions to the cv2 trackbars
        """
        type = {'Exposure':200000, 'Gain':200}
        for i in range(len(self.cameras)):
            params = create_camera_control_functions(self.cameras[i])
            for key, val in type.items():
                cv2.createTrackbar(f'{key}_{i}', cv2_window_name, 0, val, params[key])
                
    def _grab_results(self) -> list:
        grabResults = []
        for cam in self.cameras:
            grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                grabResults.append(grabResult)
        return grabResults
    
    def _grab_release(self, grabResults: list) -> None:
        for grabResult in grabResults:
            grabResult.Release()
    
    def _flip(self) -> None:
        cv2.namedWindow('Acquisition', cv2.WINDOW_NORMAL)
        width = sum([i.Width.GetValue() for i in self.cameras])
        height = max([i.Height.GetValue() for i in self.cameras])
        self._camera_params_setting(cv2_window_name='Acquisition')
        cv2.resizeWindow('Acquisition', int(width//(2.5)), int(height//(2.5)))
        self._start_grabbing()
        while all(cam.IsGrabbing() for cam in self.cameras):
            grabResults = self._grab_results()
            imgs = [grabResult.GetArray() for grabResult in grabResults]
            if len(grabResults) > 1:
                combined_image = imgs[0]
                for img in imgs[1:]:
                    combined_image = self._combine_images(combined_image, img) 
                cv2.imshow('Acquisition', combined_image)
                key = cv2.waitKey(1)
                if key == ord('f'):  # 'f' key to flip
                    self.flip = not self.flip
                    print('Image logic flipped.')
                elif key == 27:  # Escape key
                    break
        self._grab_release(grabResults)
        self._stop_grabbing()
        cv2.destroyAllWindows()

    def initialize(self, timeout:int=10) -> None:
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

    def print_all_camera_status(self) -> None:
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
        
    def _check_cameras_state(self, timeout: int=10) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if all(camera.GevIEEE1588Status.Value in ['Slave', 'Master'] for camera in self.cameras):
                for i, camera in enumerate(self.cameras):
                    if camera.GevIEEE1588Status.Value == 'Master':
                        self.master = i
                return True
            time.sleep(0.5)
        raise TimeoutError("Cameras did not become ready within the timeout period.")
    
    def _max_time_difference(self, timestamps: list) -> int:
        return max(timestamps) - min(timestamps)

    def check_sync_status(self) -> float:
        # check PTP offset
        for camera in self.cameras:
            camera.GevIEEE1588DataSetLatch.Execute()
        return [c.GevIEEE1588OffsetFromMaster.Value for c in self.cameras if c.GevIEEE1588Status.Value == 'Slave'] 

    def synchronization(self, threshold : int=300, timeout: int=20) -> list:
        self._check_cameras_state()
        print('Waiting for PTP time synchronization...')
        offset = float('inf')
        records = []
        start_time = time.time()
        while time.time() - start_time < timeout:
            offset = max(self.check_sync_status())
            records.append(offset)
            print(offset)
            offset = abs(offset)
            if offset < threshold: 
                print("Cameras synchronized.")
                return records
            time.sleep(1)
        raise TimeoutError("Cameras did not synchronize within the timeout period.")
    
    def schedule_action_command(self, scheduled_time: int) -> np.ndarray:
        self.cameras[self.master].GevTimestampControlLatch.Execute() # Get the current timestamp from the master device
        current_time = self.cameras[self.master].GevTimestampValue.Value
        scheduled_time += current_time  # delay for action command (in nanoseconds)
        self._start_grabbing()
        results = self.GigE_TL.IssueScheduledActionCommandNoWait(self.action_key, self.group_key, self.group_mask,
                                                                 scheduled_time, self.boardcast_ip)
        print(f"Scheduled command issued, retriving image...")
        grabResults = self._grab_results()
        combined_image = None
        if len(grabResults) > 1:
            imgs = [grabResult.GetArray() for grabResult in grabResults]
            timedif = self._max_time_difference([grabResult.TimeStamp for grabResult in grabResults])
            if timedif < 1000: # nanoseconds
                combined_image = imgs[0]
                for img in imgs[1:]:
                    combined_image = self._combine_images(combined_image, img)
                print(f"Image retrived.")
        self._grab_release(grabResults)
        self._stop_grabbing() 
        return combined_image
    
    # def perodically_scheduled_action_command(self, total_num: int = 10, schedule_time: int = 1000) -> np.ndarray:
    #     for _ in range(total_num):
    #         image = self.schedule_action_command(int(schedule_time * 1e6))
    #         if image is not None:
    #             return image  
    
    def end(self) -> None:
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



    # def schedule_action_command(self, scheduled_time: int) -> np.ndarray:
    #     self.cameras[0].GevTimestampControlLatch.Execute() # Get the current timestamp from the camera
    #     current_time = self.cameras[0].GevTimestampValue.Value
    #     scheduled_time = current_time + scheduled_time  # Define the delay for action command (in nanoseconds)
    #     self._start_grabbing()
    #     # Issue the scheduled action command
    #     results = self.GigE_TL.IssueScheduledActionCommandNoWait(self.action_key, self.group_key, self.group_mask,
    #                                                              scheduled_time, self.boardcast_ip)
    #     print(f"Scheduled command issued at {int(scheduled_time//1e6)}ms later, retriving image...")
        
    #     grabResult0 = self.cameras[0].RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
    #     grabResult1 = self.cameras[1].RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
    #     if grabResult0.GrabSucceeded() & grabResult1.GrabSucceeded():
    #         im0 = grabResult0.GetArray()
    #         im1 = grabResult1.GetArray()
    #         t0 = grabResult0.TimeStamp
    #         t1 = grabResult1.TimeStamp
    #         timedif = self.max_time_difference([t0, t1])
    #         if timedif < 1000:
    #             combined_image = self._combine_images(im0, im1)
    #             print("Image retrived.")
    #         else: combined_image = None
            
    #     grabResult0.Release()
    #     grabResult1.Release()
    #     self._stop_grabbing() 
    #     return combined_image