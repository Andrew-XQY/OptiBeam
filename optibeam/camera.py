from pypylon import pylon
import numpy as np
import cv2
import time
from .utils import timeout, print_underscore


def create_camera_control_functions(camera):
    """
    Callback functions to set exposure and gain using closures
    
    Args:
        camera: pylon.InstantCamera
        
    Returns:
        dict
    """
    def set_exposure(val):
        camera.ExposureTimeAbs.Value = val
    def set_gain(val):
        camera.GainRaw.Value = val
    return {'Exposure' : set_exposure, 'Gain' : set_gain}
    
class MultiBaslerCameraManager:
    """
    This is the class mainly for manage #two Basler cameras for fiber optics speckle and ground truth image acquisition.
    
    Attributes:
        cameras: list
        flip: bool
        master: int
        tlFactory: pylon.TlFactory
        GigE_TL: pylon.Tl
        action_key: int
        group_key: int
        group_mask: int
        boardcast_ip: str
        grab_timeout: int
        
    Args:
        params: dict
    """
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
        
    def _start_grabbing(self) -> None:
        """
        Start grabbing images from all cameras
        """
        for cam in self.cameras:
            cam.StartGrabbing() # pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser
    
    def _stop_grabbing(self) -> None:
        """
        Stop grabbing images from all cameras
        """
        for cam in self.cameras:
            cam.StopGrabbing()
            
    def _combine_images(self, im0 :np.ndarray, im1 :np.ndarray) -> np.ndarray:
        """
        Combine two images horizontally side by side, flip the order if self.flip is True
        
        Args:
            im0: np.ndarray
            im1: np.ndarray
            
        Returns:
            np.ndarray
        """
        return np.hstack((im0, im1) if not self.flip else (im1, im0))

    def _camera_params_setting(self, cv2_window_name: str) -> None:
        """
        Mount the camera control functions to the cv2 trackbars, each camera has its own trackbars for exposure and gain
        
        Args:
            cv2_window_name: str
            
        Returns:
            None
        """
        type = {'Exposure':100000, 'Gain':360} # exposure time in microseconds
        for i in range(len(self.cameras)):
            params = create_camera_control_functions(self.cameras[i])
            for key, val in type.items():
                cv2.createTrackbar(f'{key}_{i}', cv2_window_name, 0, val, params[key])
                
    def _grab_results(self) -> list:
        """
        Retrieve the grab results from all cameras, return a list of grab results
        
        Args:
            None
            
        Returns:
            list
        """
        grabResults = []
        for cam in self.cameras:
            grabResult = cam.RetrieveResult(self.grab_timeout, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                grabResults.append(grabResult)
        return grabResults
    
    def _grab_release(self, grabResults: list) -> None:
        """
        Release all camera grabs before the next grab
        
        Args:
            grabResults: list
            
        Returns:
            None
        """
        for grabResult in grabResults:
            grabResult.Release()
            
    def _plot_image_label(self, img: np.ndarray, label: str) -> None:
        """
        Plot a text label on the image
        
        Args:
            img: np.ndarray
            label: str
        
        Returns:
            None
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(label), (10, 50), font, 2, (255, 255, 255), 2)
        
    def _plot_max_pixel(self, img: np.ndarray) -> None:
        """
        Get the max pixel value of the image and plot on the image
        
        Args:
            img: np.ndarray
        
        Returns:
            None
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        max_pixel = 'Max pixel value: ' + str(np.max(img))
        cv2.putText(img, max_pixel, (10, 100), font, 2, (255, 255, 255), 2)
    
    def _flip_order(self) -> None:
        """
        Press 'f' key to flip the image logic (the order when combining images). 'speckle' and 'ground truth' or 'ground truth' and 'speckle'
        It set the flag variable self.flip to True or False and influence the function _combine_images
        this function will also create a window to display the images and allow the user to adjust the exposure and gain of the cameras.
        The callibration process will be done in this step.
    
        Args:
            None
        
        Returns:
            None
        """
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
                    self._plot_image_label(combined_image, 'Ground Truth')
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
        PTP configuration for each camera, configurations before starting synchronization using IEEE1588
        
        Args:
            cam: pylon.InstantCamera
        
        Returns:
            None
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
        
        
    @timeout(500)
    def initialize(self) -> None:
        """
        This function will first detect all cameras and initialize them (set a basic aquisition parameters). 
        Then Call _flip_order function to check if the user wants to flip the order of the images. 
        Finally, it will set up the PTP configuration for each camera and print the status of all cameras.
        
        Args:
            None
        
        Returns:
            None
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
        """
        Print the status of all cameras
        
        Args:
            None
        
        Returns:
            None
        """
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
        """
        Check if all cameras are configured in PTP state, they should have an attribute of GevIEEE1588Status.Value in either 'Master' or 'Slave' if success.
        In this step, the self.master is also detected and set.
        
        Args:
            None
        
        Returns:
            bool
        """
        while True:
            if all(camera.GevIEEE1588Status.Value in ['Slave', 'Master'] for camera in self.cameras):
                for i, camera in enumerate(self.cameras):
                    if camera.GevIEEE1588Status.Value == 'Master':
                        self.master = i
                return True
            time.sleep(0.5)
        
    def _max_time_difference(self, timestamps: list) -> int:
        """
        Calculate the maximum time difference in a timestamps list
        
        Args:
            timestamps: list
            
        Returns:
            int
        """
        return max(timestamps) - min(timestamps)

    def check_sync_status(self) -> float:
        """
        Check the PTP offset of all slave cameras, return a list of slave cameras clock offset with respect to the master camera
        
        Args:
            None
        
        Returns:
            list
        """
        for camera in self.cameras:
            camera.GevIEEE1588DataSetLatch.Execute()
        return [c.GevIEEE1588OffsetFromMaster.Value for c in self.cameras if c.GevIEEE1588Status.Value == 'Slave'] 
    
    @timeout(20)
    def synchronization(self, threshold : int=300) -> list:
        """
        Start the synchronization of all cameras in the network, the threshold is the maximum time difference allowed between cameras.
        The function will print the offset of all slave cameras every second until the offset is less than the threshold.
        
        Args:
            threshold: int
        
        Returns:
            list
        """
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
        """
        This is the function to issue a scheduled action command to all cameras in the network, the scheduled time is the delay in nanoseconds.
        should be only called after cameras are set to be in PTP mode and the synchronization is done.
        
        Args:
            scheduled_time: int
            
        Returns:
            np.ndarray: combined image
        """
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
        """
        API for metadata class to get the camera configurations
        
        Args:
            None
            
        Returns:
            dict: camera configurations
        """
        config = {}
        c1, c2 = 0, 1
        if self.flip:
            c1, c2 = 1, 0
            
        config['ground_truth_camera_exposure'] = self.cameras[c1].ExposureTimeAbs.Value 
        config['ground_truth_camera_gain'] = self.cameras[c1].GainRaw.Value
        config['ground_truth_camera_sn'] = self.cameras[c1].GetDeviceInfo().GetSerialNumber()
        config['ground_truth_camera_model'] = self.cameras[c1].GetDeviceInfo().GetModelName()
        config['speckle_camera_exposure'] = self.cameras[c2].ExposureTimeAbs.Value
        config['speckle_camera_gain'] = self.cameras[c2].GainRaw.Value
        config['speckle_camera_sn'] = self.cameras[c2].GetDeviceInfo().GetSerialNumber()
        config['speckle_camera_model'] = self.cameras[c2].GetDeviceInfo().GetModelName()
        return config
    
    def end(self) -> None:
        """
        After the acquisition is done, call this function to close the cameras and terminate the grab.
        
        Args:
            None
            
        Returns:
            None
        """
        self._stop_grabbing()
        for cam in self.cameras:
            cam.Close()
        print("Camera closed, grab terminated.")