from conf import *
from pypylon import pylon
from datetime import datetime
from ALP4 import *
import numpy as np
import os
import cv2


class CameraCapture:
    
    def print_camera_info(self):
        # Get the camera info and print it
        info = self.camera.GetDeviceInfo()
        print(''.join(['-']*50))
        print(f"Camera information: {info.GetModelName()}")
        print(f"Camera serial number: {info.GetSerialNumber()}")
        print(f"Camera device version: {info.GetDeviceVersion()}")
        print(f"Camera device class: {info.GetDeviceClass()}")
        print(''.join(['-']*50))
    
    
    def __init__(self, camera_index=0, exposure_us=40000):
        """
        Initialize camera capture
        
        Args:
            camera_index: int - which camera to use (0 for first, 1 for second, etc.)
            exposure_us: int - exposure time in microseconds
        """
        self.camera = None
        self.camera_index = camera_index
        self.exposure_us = exposure_us
        self.open_camera()
        self.converter = pylon.ImageFormatConverter()
        # Setting the converter to output mono8 images
        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        
        # Ensure the camera exposure, gain, and gamma are set to manual mode before adjusting
        self.camera.ExposureAuto.SetValue('Off')  # Turn off auto exposure
        self.camera.GainAuto.SetValue('Off')      # Turn off auto gain
        self.camera.GammaEnable.SetValue(True)    # Enable gamma correction if supported
        
        # Set camera parameters
        self.camera.ExposureTimeRaw.SetValue(exposure_us)  # Exposure in microseconds
        self.camera.GainRaw.SetValue(0)            # Set gain to 0
        self.camera.Gamma.SetValue(1.0)            # Set gamma value to 1.0
        
        self.print_camera_info()
        
    def open_camera(self):
        if self.camera is not None:
            try:
                self.camera.Close()
            except Exception:
                pass
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if self.camera_index < len(devices):
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[self.camera_index]))
        else:
            raise ValueError(f"Camera index {self.camera_index} not found. Available cameras: {len(devices)}")
        self.camera.Open()
    
    def capture_single_image(self):
        """Capture a single image from the camera"""
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        try:
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Convert to numpy array
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                grabResult.Release()
                return img
            else:
                grabResult.Release()
                raise RuntimeError("Image grab failed")
        finally:
            self.camera.StopGrabbing()
    
    def get_camera_parameters(self):
        """Get camera core parameters that affect the image"""
        exposure_us = self.camera.ExposureTimeRaw.Value
        exposure_ms = exposure_us / 1000.0
        return {
            'Exposure (us)': f'{exposure_us}',
            'Exposure (ms)': f'{exposure_ms:.2f}',
            'Gain': f'{self.camera.GainRaw.Value}',
            'Gamma': f'{self.camera.Gamma.Value:.2f}',
            'Camera Index': f'{self.camera_index}',
            'Model': f'{self.camera.GetDeviceInfo().GetModelName()}',
            'Serial': f'{self.camera.GetDeviceInfo().GetSerialNumber()}'
        }

    def close(self):
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        self.camera.Close()


def analyze_frame_properties(image, normalize_range=(0, 100)):
    """
    Analyze frame properties and normalize them to a specified range
    
    Args:
        image: np.ndarray
        normalize_range: tuple, default (0, 100)
    
    Returns:
        dict: frame properties with normalized values
    """
    max_pixel = np.max(image)
    total_sum = np.sum(image, dtype=np.float64)
    
    # Normalize to the specified range
    min_val, max_val = normalize_range
    
    # Normalize max pixel value (assuming 8-bit or 16-bit images)
    max_possible = 255 if image.dtype == np.uint8 else 65535
    normalized_max = min_val + (max_pixel / max_possible) * (max_val - min_val)
    
    # Normalize total sum (based on theoretical maximum)
    max_possible_sum = image.size * max_possible
    normalized_sum = min_val + (total_sum / max_possible_sum) * (max_val - min_val)
    
    properties = {
        'Max Pixel Value': f'{normalized_max:.2f}',
        'Total Sum': f'{normalized_sum:.2f}'
    }
    
    return properties


def add_text_to_image(image, camera_params, frame_properties, text_scale=1.0):
    """
    Add camera parameters and frame properties as text overlay on the image
    
    Args:
        image: np.ndarray - input grayscale image
        camera_params: dict - camera parameters
        frame_properties: dict - frame analysis properties
        text_scale: float - text scaling factor
    
    Returns:
        np.ndarray - image with text overlay (converted to BGR for colored text)
    """
    # Convert grayscale to BGR so we can add colored text
    if len(image.shape) == 2:
        img_bgr = np.stack([image, image, image], axis=2)
    else:
        img_bgr = image.copy()
    
    # Overlay frame properties (top left, white)
    y_position = int(30 * text_scale)
    line_spacing = int(35 * text_scale)
    thickness = max(1, int(2 * text_scale))
    
    for key, value in frame_properties.items():
        text = f'{key}: {value}'
        # Add white text with black outline for better visibility
        cv2.putText(img_bgr, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                   text_scale, (0, 0, 0), thickness + 2)  # Black outline
        cv2.putText(img_bgr, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                   text_scale, (255, 255, 255), thickness)  # White text
        y_position += line_spacing
    
    # Overlay camera parameters (bottom right, blue)
    y_position = img_bgr.shape[0] - int(20 * text_scale)
    for key, value in reversed(list(camera_params.items())):
        text = f'{key}: {value}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)[0]
        x_position = img_bgr.shape[1] - text_size[0] - 10
        # Add blue text with black outline for better visibility
        cv2.putText(img_bgr, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                   text_scale, (0, 0, 0), thickness + 2)  # Black outline
        cv2.putText(img_bgr, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                   text_scale, (255, 0, 0), thickness)  # Blue text
        y_position -= line_spacing
    
    return img_bgr


def image_resize(img, scale_percent):
    """
    Resize image by a scaling factor
    
    Args:
        img: np.ndarray - input image
        scale_percent: float - scaling factor (e.g., 0.6 for 60%)
    
    Returns:
        np.ndarray - resized image
    """
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv2.resize(img, dim)


def save_image_with_info(img, camera_params, frame_properties, save_dir, text_scale=1.0):
    """
    Save image with text overlay
    
    Args:
        img: np.ndarray - raw image
        camera_params: dict - camera parameters
        frame_properties: dict - frame properties
        save_dir: str - directory to save the image
        text_scale: float - text scaling factor
    
    Returns:
        str - filename of saved image
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Add text overlay to image
    img_with_info = add_text_to_image(img, camera_params, frame_properties, text_scale)
    
    # Generate timestamp filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(save_dir, f"{timestamp}.png")
    
    # Save image
    cv2.imwrite(filename, img_with_info)
    return filename


def live_preview_and_capture(save_dir, camera_index=0, initial_exposure_us=40000, text_scale=1.0, scale_factor=0.6, update_rate_hz=1.0, dmd_device=None):
    """
    Display live preview with CV2 window, updating at specified rate, with exposure control
    
    Args:
        save_dir: str - directory to save the image
        camera_index: int - which camera to use (0 for first, 1 for second, etc.)
        initial_exposure_us: int - initial exposure time in microseconds
        text_scale: float - text scaling factor for overlays
        scale_factor: float - rescale ratio for display window (default 0.6)
        update_rate_hz: float - update rate in Hz (frames per second), default 1.0
        dmd_device: DMD device object (optional) - if provided, will be used for displaying pattern
    """
    import time
    import tkinter as tk
    from tkinter import messagebox
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate wait time based on update rate
    wait_time_ms = int(1000 / update_rate_hz)  # Convert Hz to milliseconds
    
    # Initialize camera
    camera_capture = CameraCapture(camera_index=camera_index, exposure_us=initial_exposure_us)
    
    # Create CV2 window
    cv2.namedWindow('Camera Live Preview')
    
    # Create a simple Tkinter window for exposure input
    def create_exposure_input_window():
        input_window = tk.Tk()
        input_window.title("Set Exposure Time")
        input_window.geometry("350x120")
        
        tk.Label(input_window, text="Exposure Time (microseconds):", font=("Arial", 10)).pack(pady=10)
        
        exposure_entry = tk.Entry(input_window, font=("Arial", 12), width=15)
        exposure_entry.insert(0, str(initial_exposure_us))
        exposure_entry.pack(pady=5)
        
        def set_exposure():
            try:
                new_exposure_us = int(exposure_entry.get())
                if new_exposure_us <= 0:
                    messagebox.showerror("Error", "Exposure must be positive!")
                    return
                camera_capture.camera.ExposureTimeRaw.SetValue(new_exposure_us)
                print(f"Exposure updated to {new_exposure_us} us ({new_exposure_us/1000:.2f} ms)")
                messagebox.showinfo("Success", f"Exposure set to {new_exposure_us} µs")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to set exposure: {e}")
        
        set_button = tk.Button(input_window, text="Set Exposure", command=set_exposure, 
                              font=("Arial", 10), bg="lightblue", width=15)
        set_button.pack(pady=10)
        
        # Make window stay on top
        input_window.attributes('-topmost', True)
        
        return input_window
    
    # Create exposure input window
    exposure_window = create_exposure_input_window()
    
    print("Live preview started. Press 's' to save, ESC to exit.")
    print(f"Update rate: {update_rate_hz} Hz ({1/update_rate_hz:.2f} seconds per frame)")
    print(f"Display scale factor: {scale_factor}")
    
    try:
        while True:
            start_time = time.time()
            
            # Update Tkinter window
            try:
                exposure_window.update()
            except tk.TclError:
                # Window was closed
                print("Exposure input window closed. Exiting...")
                break
            
            # Capture image
            img = camera_capture.capture_single_image()
            
            # Get camera parameters and frame properties
            camera_params = camera_capture.get_camera_parameters()
            frame_properties = analyze_frame_properties(img)
            
            # Add text overlay to image for display
            img_with_info = add_text_to_image(img, camera_params, frame_properties, text_scale)
            
            # Resize for display
            img_display = image_resize(img_with_info, scale_factor)
            
            # Display the image
            cv2.imshow('Camera Live Preview', img_display)
            
            # Wait for key press based on update rate
            key = cv2.waitKey(wait_time_ms)
            
            if key == 27:  # ESC key to exit
                print("Exiting...")
                break
            elif key == ord('s') or key == ord('S'):  # 's' or 'S' key to save
                # Save the original full-resolution image with info
                filename = save_image_with_info(img, camera_params, frame_properties, save_dir, text_scale)
                print(f"Image saved: {filename}")
                
                # Print summary
                print("\nImage Properties:")
                for k, v in frame_properties.items():
                    print(f"  {k}: {v}")
                print("\nCamera Parameters:")
                for k, v in camera_params.items():
                    print(f"  {k}: {v}")
                print()
            
            # Ensure we're updating at the specified rate
            elapsed = time.time() - start_time
            target_interval = 1.0 / update_rate_hz
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
                
    finally:
        camera_capture.close()
        cv2.destroyAllWindows()
        try:
            exposure_window.destroy()
        except:
            pass


# DMD constants
DMD_DIM = 1024


if __name__ == "__main__":
    # Configuration
    CAMERA_INDEX = 1           # Which camera to use (0 for first, 1 for second, etc.)
    INITIAL_EXPOSURE_US = 250000  # Initial exposure time in microseconds (250000 us = 250 ms)
    SAVE_DIR = 'C:\\Users\\qiyuanxu\\Desktop\\'  # Directory to save images
    TEXT_SCALE = 1.0           # Text scaling factor for overlays
    SCALE_FACTOR = 0.7         # Display window rescale ratio (0.7 = 70% of original size)
    UPDATE_RATE_HZ = 1.0       # Update rate in Hz (frames per second), default 1.0 Hz = 1 frame/second
    
    # Try to initialize and use the DMD; if it fails, continue in camera-only mode.
    DMD = None
    try:
        DMD = dmd.ViALUXDMD(ALP4(version='4.3'))
        # Generate the inverted upward arrow pattern
        calibration_img = simulation.generate_inverted_upward_arrow(intesity=255)
        
        # Scale up the pattern to DMD dimensions
        calibration_img = simulation.macro_pixel(calibration_img, size=int(DMD_DIM/calibration_img.shape[0])) 
        calibration_img = dmd.dmd_img_adjustment(calibration_img, DMD_DIM, angle=DMD_ROTATION_ANGLE)
        
        DMD.display_image(calibration_img)  # Display pattern on DMD
        print("DMD initialized successfully with inverted upward arrow pattern.")
    except Exception as e:
        print(f"Warning: DMD initialization or display failed ({e}). Running in camera-only mode.")
    
    # Start live preview with capture capability
    live_preview_and_capture(
        save_dir=SAVE_DIR,
        camera_index=CAMERA_INDEX,
        initial_exposure_us=INITIAL_EXPOSURE_US,
        text_scale=TEXT_SCALE,
        scale_factor=SCALE_FACTOR,
        update_rate_hz=UPDATE_RATE_HZ,
        dmd_device=DMD
    )
