from pypylon import pylon
import numpy as np
import cv2
import time
from .utils import timeout, print_underscore


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

    def __init__(self, params={"grab_timeout" : 5000, "action_key" : 0x1, "group_key" : 0x1,
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
        self.grab_timeout = params.get("grab_timeout")
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
        type = {'Exposure':200000, 'Gain':400}
        for i in range(len(self.cameras)):
            params = create_camera_control_functions(self.cameras[i])
            for key, val in type.items():
                cv2.createTrackbar(f'{key}_{i}', cv2_window_name, 0, val, params[key])
                
    def _grab_results(self) -> list:
        grabResults = []
        for cam in self.cameras:
            grabResult = cam.RetrieveResult(self.grab_timeout, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                grabResults.append(grabResult)
        return grabResults
    
    def _grab_release(self, grabResults: list) -> None:
        for grabResult in grabResults:
            grabResult.Release()
            
    def _plot_max_pixel(self, img: np.ndarray) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        max_pixel = 'Max pixel value: ' + str(np.max(img))
        cv2.putText(img, max_pixel, (10, 50), font, 2, (255, 255, 255), 2)
    
    def _flip_order(self) -> None:
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
                self._plot_max_pixel(combined_image)
                for img in imgs[1:]:
                    self._plot_max_pixel(img)
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
        
    def _ptp_setup(self, cam: pylon.InstantCamera):
        """
        PTP configuration for each camera
        """
        cam.AcquisitionFrameRateEnable.Value = False
        cam.GevIEEE1588.Value = True
        cam.AcquisitionMode.SetValue("SingleFrame") # SingleFrame Continuous
        cam.TriggerMode.SetValue("On")
        cam.TriggerSource.SetValue("Action1")
        cam.TriggerSelector.SetValue('FrameStart')
        cam.ActionDeviceKey.SetValue(self.action_key)
        cam.ActionGroupKey.SetValue(self.group_key)
        cam.ActionGroupMask.SetValue(self.group_mask)
        
    @timeout(10)
    def initialize(self) -> None:
        """
        detect all cameras and initialize them
        """
        while True:
            devices = self.tlFactory.EnumerateDevices()
            if len(devices) >= 2:
                break
            time.sleep(0.5)
        
        for i in range(len(devices)):  
            camera = pylon.InstantCamera(self.tlFactory.CreateDevice(devices[i]))
            camera.Open()
            camera.AcquisitionFrameRateEnable.Value = True
            camera.AcquisitionFrameRateAbs.Value = 20.0
            self.cameras.append(camera)
        self._flip_order()
        
        for i in self.cameras:  # prepare for PTP and scheduled action command
            self._ptp_setup(i)
        self.print_all_camera_status()

    @print_underscore
    def print_all_camera_status(self) -> None:
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

    @timeout(10)
    def _check_cameras_ptp_state(self) -> bool:
        while True:
            if all(camera.GevIEEE1588Status.Value in ['Slave', 'Master'] for camera in self.cameras):
                for i, camera in enumerate(self.cameras):
                    if camera.GevIEEE1588Status.Value == 'Master':
                        self.master = i
                return True
            time.sleep(0.5)
        
    def _max_time_difference(self, timestamps: list) -> int:
        return max(timestamps) - min(timestamps)

    def check_sync_status(self) -> float:
        # check PTP offset
        for camera in self.cameras:
            camera.GevIEEE1588DataSetLatch.Execute()
        return [c.GevIEEE1588OffsetFromMaster.Value for c in self.cameras if c.GevIEEE1588Status.Value == 'Slave'] 
    
    @timeout(20)
    def synchronization(self, threshold : int=300) -> list:
        self._check_cameras_ptp_state()
        print('Waiting for PTP time synchronization...')
        offset = float('inf')
        records = []
        while True:
            offset = max(self.check_sync_status())
            records.append(offset)
            print(offset)
            offset = abs(offset)
            if offset < threshold: 
                print("Cameras synchronized.")
                return records
            time.sleep(1)
    
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
    
    def get_metadata(self) -> dict:
        config = {}
        c1, c2 = 0, 1
        if self.flip:
            c1, c2 = 1, 0
        config['ground_truth_camera_exposure'] = self.cameras[c1].ExposureTimeAbs.Value 
        config['ground_truth_camera_gain'] = self.cameras[c1].GainRaw.Value
        config['ground_truth_camera_sn'] = self.cameras[c1].GetDeviceInfo().GetSerialNumber()
        config['speckle_camera_exposure'] = self.cameras[c2].ExposureTimeAbs.Value
        config['speckle_camera_gain'] = self.cameras[c2].GainRaw.Value
        config['speckle_camera_sn'] = self.cameras[c2].GetDeviceInfo().GetSerialNumber()
        return config
    
    def end(self) -> None:
        self._stop_grabbing()
        for cam in self.cameras:
            cam.Close()
        print("Camera closed, grab terminated.")
    
    