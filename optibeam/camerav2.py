"""
Multi-camera manager v2 - Adapted for data_collection_v2.py
- Removed CV2 trackbar/slider logic
- No interactive configuration window
- Direct parameter setting via methods
- Cleaner API for automated data collection
"""
from pypylon import pylon
import numpy as np
import cv2
import time
from .utils import timeout, print_underscore


class MultiBaslerCameraManager:
    """
    Multi-camera manager for Basler cameras (v2)
    Designed for automated data collection without interactive UI elements
    
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
    def __init__(self, params={"grab_timeout": 5000, "action_key": 0x1, "group_key": 0x1,
                               "group_mask": 0xffffffff, "boardcast_ip": "255.255.255.255"}):
        self.cameras = []
        self.flip = False
        self.master = None
        self.tlFactory = pylon.TlFactory.GetInstance()
        self.GigE_TL = self.tlFactory.CreateTl('BaslerGigE')
        self.action_key = params.get("action_key")
        self.group_key = params.get("group_key")
        self.group_mask = params.get("group_mask")
        self.boardcast_ip = params.get("boardcast_ip")
        self.grab_timeout = params.get("grab_timeout")
        
    def _start_grabbing(self) -> None:
        """Start grabbing images from all cameras"""
        for cam in self.cameras:
            cam.StartGrabbing()
    
    def _stop_grabbing(self) -> None:
        """Stop grabbing images from all cameras"""
        for cam in self.cameras:
            cam.StopGrabbing()
            
    def _combine_images(self, im0: np.ndarray, im1: np.ndarray) -> np.ndarray:
        """
        Combine two images horizontally side by side
        Flip order if self.flip is True
        """
        return np.hstack((im0, im1) if not self.flip else (im1, im0))

    def _grab_results(self) -> list:
        """Retrieve grab results from all cameras"""
        grabResults = []
        for cam in self.cameras:
            grabResult = cam.RetrieveResult(self.grab_timeout, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                grabResults.append(grabResult)
        return grabResults
    
    def _grab_release(self, grabResults: list) -> None:
        """Release all camera grabs before the next grab"""
        for grabResult in grabResults:
            grabResult.Release()
    
    def _ptp_setup(self, cam: pylon.InstantCamera):
        """PTP configuration for each camera"""
        cam.AcquisitionFrameRateEnable.Value = False
        cam.GevIEEE1588.Value = True
        cam.AcquisitionMode.SetValue("SingleFrame")
        cam.TriggerMode.SetValue("On")
        cam.TriggerSource.SetValue("Action1")
        cam.TriggerSelector.SetValue('FrameStart')
        cam.ActionDeviceKey.SetValue(self.action_key)
        cam.ActionGroupKey.SetValue(self.group_key)
        cam.ActionGroupMask.SetValue(self.group_mask)
        
    def _initialize_cams(self):
        """Initialize all detected cameras"""
        while True:
            devices = self.tlFactory.EnumerateDevices()
            if len(devices) >= 2:
                break
            time.sleep(0.5)
        
        for i in range(len(devices)):  
            camera = pylon.InstantCamera(self.tlFactory.CreateDevice(devices[i]))
            camera.Open()
            
            # Disable auto exposure for manual control
            camera.ExposureAuto.Value = 'Off'
            print(f"Camera {i}: ExposureAuto current value: {camera.ExposureAuto.Value}")
            
            # Disable auto gain for manual control
            camera.GainAuto.Value = 'Off'
            print(f"Camera {i}: GainAuto current value: {camera.GainAuto.Value}")
            
            camera.AcquisitionFrameRateEnable.Value = True
            camera.AcquisitionFrameRateAbs.Value = 20.0
            self.cameras.append(camera)
    
    def set_camera_exposure(self, camera_index: int, exposure_us: int) -> None:
        """
        Set exposure time for a specific camera
        
        Args:
            camera_index: int - camera index (0, 1, etc.)
            exposure_us: int - exposure time in microseconds
        """
        if camera_index < len(self.cameras):
            min_exposure_time = 50  # Minimum safe exposure time (µs)
            self.cameras[camera_index].ExposureTimeAbs.Value = max(min_exposure_time, exposure_us)
            print(f"Camera {camera_index} exposure set to {exposure_us} µs")
    
    def set_camera_gain(self, camera_index: int, gain: int) -> None:
        """
        Set gain for a specific camera
        
        Args:
            camera_index: int - camera index (0, 1, etc.)
            gain: int - gain value
        """
        if camera_index < len(self.cameras):
            self.cameras[camera_index].GainRaw.Value = gain
            print(f"Camera {camera_index} gain set to {gain}")
    
    def set_all_cameras_exposure(self, exposure_us: int) -> None:
        """Set same exposure time for all cameras"""
        for i in range(len(self.cameras)):
            self.set_camera_exposure(i, exposure_us)
    
    def set_all_cameras_gain(self, gain: int) -> None:
        """Set same gain for all cameras"""
        for i in range(len(self.cameras)):
            self.set_camera_gain(i, gain)
    
    def set_flip(self, flip: bool) -> None:
        """
        Set the flip flag to change image combination order
        
        Args:
            flip: bool - True to flip order, False for default order
        """
        self.flip = flip
        print(f"Image combination order flip set to: {flip}")
    
    def _setup_configuration_window(self, update_rate_hz: float = 1.0, scale_factor: float = 0.6, 
                                   text_scale: float = 0.8, save_dir: str = None, window_scale: float = 0.6) -> None:
        """
        Open configuration window with live preview and parameter controls
        Similar to filter_test.py but for two cameras
        
        Args:
            update_rate_hz: float - preview update rate in Hz
            scale_factor: float - display scale factor
            text_scale: float - text overlay scale
            save_dir: str - directory to save images when 's' is pressed
            window_scale: float - CV2 window scale ratio (e.g., 0.4, 0.7, 1.0, 1.5)
        """
        import time
        import tkinter as tk
        from tkinter import messagebox
        import os
        
        # Create Tkinter parameter window
        def create_param_window():
            param_win = tk.Tk()
            param_win.title("Camera Parameters")
            param_win.geometry("400x320")
            
            # Camera 0 Exposure
            tk.Label(param_win, text="Camera 0 Exposure (us):", font=("Arial", 10)).pack(pady=5)
            exp0_entry = tk.Entry(param_win, font=("Arial", 11), width=20)
            exp0_entry.insert(0, str(int(self.cameras[0].ExposureTimeAbs.Value)))
            exp0_entry.pack(pady=2)
            
            # Camera 0 Gain
            tk.Label(param_win, text="Camera 0 Gain:", font=("Arial", 10)).pack(pady=5)
            gain0_entry = tk.Entry(param_win, font=("Arial", 11), width=20)
            gain0_entry.insert(0, str(self.cameras[0].GainRaw.Value))
            gain0_entry.pack(pady=2)
            
            # Camera 1 Exposure
            tk.Label(param_win, text="Camera 1 Exposure (us):", font=("Arial", 10)).pack(pady=5)
            exp1_entry = tk.Entry(param_win, font=("Arial", 11), width=20)
            exp1_entry.insert(0, str(int(self.cameras[1].ExposureTimeAbs.Value)))
            exp1_entry.pack(pady=2)
            
            # Camera 1 Gain
            tk.Label(param_win, text="Camera 1 Gain:", font=("Arial", 10)).pack(pady=5)
            gain1_entry = tk.Entry(param_win, font=("Arial", 11), width=20)
            gain1_entry.insert(0, str(self.cameras[1].GainRaw.Value))
            gain1_entry.pack(pady=2)
            
            def apply_params():
                try:
                    exp0 = int(exp0_entry.get())
                    gain0 = int(gain0_entry.get())
                    exp1 = int(exp1_entry.get())
                    gain1 = int(gain1_entry.get())
                    
                    self.set_camera_exposure(0, exp0)
                    self.set_camera_gain(0, gain0)
                    self.set_camera_exposure(1, exp1)
                    self.set_camera_gain(1, gain1)
                    
                    messagebox.showinfo("Success", "Parameters applied!")
                except ValueError:
                    messagebox.showerror("Error", "Invalid input!")
            
            apply_btn = tk.Button(param_win, text="Apply Parameters", command=apply_params,
                                 font=("Arial", 10), bg="lightgreen", width=20)
            apply_btn.pack(pady=10)
            
            param_win.attributes('-topmost', True)
            return param_win
        
        # Helper functions from filter_test.py
        def analyze_frame_properties(image, normalize_range=(0, 100)):
            max_pixel = np.max(image)
            total_sum = np.sum(image, dtype=np.float64)
            min_val, max_val = normalize_range
            max_possible = 255 if image.dtype == np.uint8 else 65535
            normalized_max = min_val + (max_pixel / max_possible) * (max_val - min_val)
            max_possible_sum = image.size * max_possible
            normalized_sum = min_val + (total_sum / max_possible_sum) * (max_val - min_val)
            return {
                'Max Pixel Value': f'{normalized_max:.2f}',
                'Total Sum': f'{normalized_sum:.2f}'
            }
        
        def get_camera_params_display():
            c1, c2 = (0, 1) if not self.flip else (1, 0)
            exp0 = self.cameras[c1].ExposureTimeAbs.Value
            exp1 = self.cameras[c2].ExposureTimeAbs.Value
            return {
                'Cam0 Exp(us)': f'{int(exp0)}',
                'Cam0 Exp(ms)': f'{exp0/1000:.2f}',
                'Cam0 Gain': f'{self.cameras[c1].GainRaw.Value}',
                'Cam1 Exp(us)': f'{int(exp1)}',
                'Cam1 Exp(ms)': f'{exp1/1000:.2f}',
                'Cam1 Gain': f'{self.cameras[c2].GainRaw.Value}',
            }
        
        def add_text_to_image(image, camera_params, frame_properties):
            if len(image.shape) == 2:
                img_bgr = np.stack([image, image, image], axis=2)
            else:
                img_bgr = image.copy()
            
            y_position = int(30 * text_scale)
            line_spacing = int(35 * text_scale)
            thickness = max(1, int(2 * text_scale))
            
            # Frame properties (top left, white)
            for key, value in frame_properties.items():
                text = f'{key}: {value}'
                cv2.putText(img_bgr, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                           text_scale, (0, 0, 0), thickness + 2)
                cv2.putText(img_bgr, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                           text_scale, (255, 255, 255), thickness)
                y_position += line_spacing
            
            # Camera parameters (bottom right, blue)
            y_position = img_bgr.shape[0] - int(20 * text_scale)
            for key, value in reversed(list(camera_params.items())):
                text = f'{key}: {value}'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)[0]
                x_position = img_bgr.shape[1] - text_size[0] - 10
                cv2.putText(img_bgr, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                           text_scale, (0, 0, 0), thickness + 2)
                cv2.putText(img_bgr, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                           text_scale, (255, 0, 0), thickness)
                y_position -= line_spacing
            
            return img_bgr
        
        def image_resize(img, scale_percent):
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            return cv2.resize(img, (width, height))
        
        def save_image(img, camera_params, frame_properties):
            if save_dir is None:
                return
            os.makedirs(save_dir, exist_ok=True)
            img_with_info = add_text_to_image(img, camera_params, frame_properties)
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(save_dir, f"{timestamp}.png")
            cv2.imwrite(filename, img_with_info)
            print(f"Image saved: {filename}")
        
        # Create windows
        param_window = create_param_window()
        cv2.namedWindow('Camera Setup', cv2.WINDOW_NORMAL)
        
        # Get combined image size for window scaling
        # Estimate based on camera resolution
        if len(self.cameras) > 0:
            width = sum([cam.Width.GetValue() for cam in self.cameras])
            height = max([cam.Height.GetValue() for cam in self.cameras])
            cv2.resizeWindow('Camera Setup', int(width * window_scale), int(height * window_scale))
        
        print("\n" + "="*60)
        print("Camera Configuration Mode")
        print("  - Press 'f' to flip camera order")
        print("  - Press 's' to save current frame")
        print("  - Press ESC to finish setup and start data collection")
        print(f"  - Update rate: {update_rate_hz} Hz")
        print(f"  - Window scale: {window_scale}")
        print("="*60 + "\n")
        
        wait_time_ms = int(1000 / update_rate_hz)
        last_update_time = time.time()
        
        self._start_grabbing()
        try:
            while all(cam.IsGrabbing() for cam in self.cameras):
                # Update Tkinter window
                try:
                    param_window.update()
                except tk.TclError:
                    print("Parameter window closed. Exiting setup...")
                    break
                
                # Check if it's time to update
                current_time = time.time()
                if current_time - last_update_time >= (1.0 / update_rate_hz):
                    last_update_time = current_time
                    
                    # Grab images
                    grabResults = self._grab_results()
                    imgs = [grabResult.GetArray() for grabResult in grabResults]
                    
                    if len(grabResults) > 1:
                        # Combine images
                        combined_image = imgs[0]
                        for img in imgs[1:]:
                            combined_image = self._combine_images(combined_image, img)
                        
                        # Get metadata and add to image
                        camera_params = get_camera_params_display()
                        frame_properties = analyze_frame_properties(combined_image)
                        img_with_info = add_text_to_image(combined_image, camera_params, frame_properties)
                        
                        # Add order indicator
                        order_text = "Flipped" if self.flip else "Normal"
                        cv2.putText(img_with_info, f'Order: {order_text}', (10, img_with_info.shape[0] - 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), 2)
                        
                        # Resize and display
                        img_display = image_resize(img_with_info, scale_factor)
                        cv2.imshow('Camera Setup', img_display)
                    
                    self._grab_release(grabResults)
                
                # Handle key presses
                key = cv2.waitKey(wait_time_ms)
                if key == 27:  # ESC
                    print("\nSetup complete. Starting data collection...")
                    break
                elif key == ord('f'):  # Flip
                    self.flip = not self.flip
                    print(f'Camera order flipped. Current: {"Flipped" if self.flip else "Normal"}')
                elif key == ord('s') or key == ord('S'):  # Save
                    grabResults = self._grab_results()
                    imgs = [grabResult.GetArray() for grabResult in grabResults]
                    if len(grabResults) > 1:
                        combined_image = imgs[0]
                        for img in imgs[1:]:
                            combined_image = self._combine_images(combined_image, img)
                        camera_params = get_camera_params_display()
                        frame_properties = analyze_frame_properties(combined_image)
                        save_image(combined_image, camera_params, frame_properties)
                    self._grab_release(grabResults)
        
        finally:
            self._stop_grabbing()
            cv2.destroyAllWindows()
            try:
                param_window.destroy()
            except:
                pass
        
    @timeout(500)
    def initialize(self, show_config_window: bool = True, config_params: dict = None) -> None:
        """
        Initialize cameras and optionally show configuration window
        
        Args:
            show_config_window: bool - if True, opens configuration window before proceeding
            config_params: dict - parameters for config window (update_rate_hz, scale_factor, text_scale, save_dir, window_scale)
        
        Returns:
            None
        """
        self._initialize_cams()
        
        # Show configuration window if requested
        if show_config_window:
            params = config_params or {}
            self._setup_configuration_window(
                update_rate_hz=params.get('update_rate_hz', 1.0),
                scale_factor=params.get('scale_factor', 0.6),
                text_scale=params.get('text_scale', 0.8),
                save_dir=params.get('save_dir', None),
                window_scale=params.get('window_scale', 0.6)
            )
        
        # Setup PTP configuration for each camera
        for i in self.cameras:
            self._ptp_setup(i)
        
        self.print_all_camera_status()

    @print_underscore
    def print_all_camera_status(self) -> None:
        """Print the status of all cameras"""
        print(f"Number of cameras detected: {len(self.cameras)}")
        print('\n')
        for cam in self.cameras:
            info = cam.GetDeviceInfo()
            print("Using %s @ %s @ %s" % (info.GetModelName(), info.GetSerialNumber(), info.GetIpAddress()))
            print("ExposureAuto:", cam.ExposureAuto.Value)
            print("GainAuto:", cam.GainAuto.Value)
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
        Check if all cameras are in PTP state (Master or Slave)
        Sets self.master to the master camera index
        """
        while True:
            if all(camera.GevIEEE1588Status.Value in ['Slave', 'Master'] for camera in self.cameras):
                for i, camera in enumerate(self.cameras):
                    if camera.GevIEEE1588Status.Value == 'Master':
                        self.master = i
                return True
            time.sleep(0.5)
        
    def _max_time_difference(self, timestamps: list) -> int:
        """Calculate the maximum time difference in timestamps"""
        return max(timestamps) - min(timestamps)

    def check_sync_status(self) -> float:
        """
        Check PTP offset of all slave cameras
        
        Returns:
            list of offsets from master camera
        """
        for camera in self.cameras:
            camera.GevIEEE1588DataSetLatch.Execute()
        return [c.GevIEEE1588OffsetFromMaster.Value for c in self.cameras if c.GevIEEE1588Status.Value == 'Slave'] 
    
    @timeout(20)
    def synchronization(self, threshold: int = 300) -> list:
        """
        Synchronize all cameras using PTP
        
        Args:
            threshold: int - maximum time difference allowed (nanoseconds)
        
        Returns:
            list of offset records
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
        Issue a scheduled action command to all cameras
        
        Args:
            scheduled_time: int - delay in nanoseconds
            
        Returns:
            np.ndarray: combined image from all cameras
        """
        self.cameras[self.master].GevTimestampControlLatch.Execute()
        current_time = self.cameras[self.master].GevTimestampValue.Value
        scheduled_time += current_time
        self._start_grabbing()
        results = self.GigE_TL.IssueScheduledActionCommandNoWait(self.action_key, self.group_key, self.group_mask,
                                                                 scheduled_time, self.boardcast_ip)
        print(f"Scheduled command issued, retrieving image...")
        
        grabResults = self._grab_results()
        combined_image = None
        if len(grabResults) > 1:
            imgs = [grabResult.GetArray() for grabResult in grabResults]
            timedif = self._max_time_difference([grabResult.TimeStamp for grabResult in grabResults])
            if timedif < 1000:  # nanoseconds
                combined_image = imgs[0]
                for img in imgs[1:]:
                    combined_image = self._combine_images(combined_image, img)
                print(f"Image retrieved.")
                
        self._grab_release(grabResults)
        self._stop_grabbing() 
        return combined_image
    
    def free_run(self):
        """
        Free run mode - yields combined images in real time
        
        Yields:
            np.ndarray: combined image
        """
        self._start_grabbing()
        while all(cam.IsGrabbing() for cam in self.cameras):
            grabResults = self._grab_results()
            imgs = [grabResult.GetArray() for grabResult in grabResults]
            if len(grabResults) > 1:
                combined_image = imgs[0]
                for img in imgs[1:]:
                    combined_image = self._combine_images(combined_image, img) 
                yield combined_image
        self._grab_release(grabResults)
        self._stop_grabbing()
    
    def get_metadata(self) -> dict:
        """
        Get camera configurations for metadata
        
        Returns:
            dict: camera configurations including exposure, gain, etc.
        """
        config = {}
        c1, c2 = 0, 1
        if self.flip:
            c1, c2 = 1, 0
        
        # Get exposure time and gain for both cameras
        exposure_time_cam1 = self.cameras[c1].ExposureTimeAbs.Value if hasattr(self.cameras[c1], 'ExposureTimeAbs') else 0
        exposure_time_cam2 = self.cameras[c2].ExposureTimeAbs.Value if hasattr(self.cameras[c2], 'ExposureTimeAbs') else 0
        
        config['ground_truth_camera_exposure'] = exposure_time_cam1
        config['ground_truth_camera_gain'] = self.cameras[c1].GainRaw.Value
        config['ground_truth_camera_sn'] = self.cameras[c1].GetDeviceInfo().GetSerialNumber()
        config['ground_truth_camera_model'] = self.cameras[c1].GetDeviceInfo().GetModelName()
        config['speckle_camera_exposure'] = exposure_time_cam2
        config['speckle_camera_gain'] = self.cameras[c2].GainRaw.Value
        config['speckle_camera_sn'] = self.cameras[c2].GetDeviceInfo().GetSerialNumber()
        config['speckle_camera_model'] = self.cameras[c2].GetDeviceInfo().GetModelName()
        
        # Also add generic fields for easier access
        config['exposure_time'] = exposure_time_cam1  # Primary camera exposure
        config['gain'] = self.cameras[c1].GainRaw.Value
        config['gamma'] = 1.0  # Default gamma value
        
        return config
    
    def end(self) -> None:
        """
        Close all cameras and terminate grabbing
        """
        self._stop_grabbing()
        for cam in self.cameras:
            cam.Close()
        print("Camera closed, grab terminated.")
