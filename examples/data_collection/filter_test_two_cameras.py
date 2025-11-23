from conf import *
from pypylon import pylon
from datetime import datetime
from ALP4 import *
import numpy as np
import os
import cv2
from collections import deque
import time
import multiprocessing as mp
import tkinter as tk
from tkinter import messagebox


class MaxPixelBuffer:
    """Buffer to track maximum pixel value over a time window"""
    
    def __init__(self, window_seconds=10.0):
        """
        Initialize the buffer
        
        Args:
            window_seconds: float - time window in seconds (default 10.0)
        """
        self.window_seconds = window_seconds
        self.buffer = deque()  # Store (timestamp, max_pixel_value) tuples
    
    def add_value(self, max_pixel_value):
        """
        Add a new max pixel value to the buffer
        
        Args:
            max_pixel_value: float - maximum pixel value from current frame
        """
        current_time = time.time()
        self.buffer.append((current_time, max_pixel_value))
        
        # Remove old values outside the time window
        cutoff_time = current_time - self.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff_time:
            self.buffer.popleft()
    
    def get_max(self):
        """
        Get the maximum pixel value within the time window
        
        Returns:
            float - maximum pixel value in the buffer, or None if buffer is empty
        """
        if not self.buffer:
            return None
        return max(value for _, value in self.buffer)
    
    def get_buffer_duration(self):
        """
        Get the actual duration of data in the buffer
        
        Returns:
            float - duration in seconds, or 0 if buffer is empty
        """
        if len(self.buffer) < 2:
            return 0.0
        return self.buffer[-1][0] - self.buffer[0][0]


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
    
    # Overlay camera parameters (bottom right, green)
    y_position = img_bgr.shape[0] - int(20 * text_scale)
    for key, value in reversed(list(camera_params.items())):
        text = f'{key}: {value}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)[0]
        x_position = img_bgr.shape[1] - text_size[0] - 10
        # Add green text with black outline for better visibility
        cv2.putText(img_bgr, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                   text_scale, (0, 0, 0), thickness + 2)  # Black outline
        cv2.putText(img_bgr, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                   text_scale, (0, 255, 0), thickness)  # Green text
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
    camera_index = camera_params.get('Camera Index', '0')
    filename = os.path.join(save_dir, f"cam{camera_index}_{timestamp}.png")
    
    # Save image
    cv2.imwrite(filename, img_with_info)
    return filename


def camera_process(camera_index, initial_exposure_us, save_dir, text_scale, scale_factor, 
                   update_rate_hz, max_buffer_seconds):
    """
    Process function for a single camera - runs in separate process
    
    Args:
        camera_index: int - which camera to use (0 or 1)
        initial_exposure_us: int - initial exposure time in microseconds
        save_dir: str - directory to save the image
        text_scale: float - text scaling factor for overlays
        scale_factor: float - rescale ratio for display window
        update_rate_hz: float - update rate in Hz (frames per second)
        max_buffer_seconds: float - time window in seconds for max pixel tracking
    """
    # Calculate wait time based on update rate
    wait_time_ms = int(1000 / update_rate_hz)  # Convert Hz to milliseconds
    
    # Initialize camera
    camera_capture = CameraCapture(camera_index=camera_index, exposure_us=initial_exposure_us)
    
    # Initialize max pixel buffer
    max_pixel_buffer = MaxPixelBuffer(window_seconds=max_buffer_seconds)
    
    # Camera labels
    camera_label = "Fiber Camera" if camera_index == 0 else "Ground Truth Camera"
    
    # Create CV2 window with unique name
    window_name = f'{camera_label} (Cam {camera_index})'
    cv2.namedWindow(window_name)
    
    # Position windows side by side (camera 0 on left, camera 1 on right)
    window_x_position = 0 if camera_index == 0 else 800
    cv2.moveWindow(window_name, window_x_position, 100)
    
    # Create a simple Tkinter window for exposure and gain input
    def create_exposure_input_window():
        input_window = tk.Tk()
        input_window.title(f"{camera_label} - Camera Parameters")
        input_window.geometry("350x200")
        
        # Position Tkinter windows vertically stacked
        window_y_position = 100 if camera_index == 0 else 350
        input_window.geometry(f"350x200+{1200}+{window_y_position}")
        
        # Exposure Time Section
        tk.Label(input_window, text=f"{camera_label} - Exposure Time (µs):", 
                font=("Arial", 10)).pack(pady=(10, 5))
        
        exposure_entry = tk.Entry(input_window, font=("Arial", 12), width=15)
        exposure_entry.insert(0, str(initial_exposure_us))
        exposure_entry.pack(pady=5)
        
        # Gain Section
        tk.Label(input_window, text=f"{camera_label} - Gain:", 
                font=("Arial", 10)).pack(pady=(10, 5))
        
        current_gain = camera_capture.camera.GainRaw.Value
        gain_entry = tk.Entry(input_window, font=("Arial", 12), width=15)
        gain_entry.insert(0, str(current_gain))
        gain_entry.pack(pady=5)
        
        def set_parameters():
            success_messages = []
            error_occurred = False
            
            # Set Exposure
            try:
                new_exposure_us = int(exposure_entry.get())
                if new_exposure_us <= 0:
                    messagebox.showerror("Error", "Exposure must be positive!")
                    error_occurred = True
                else:
                    camera_capture.camera.ExposureTimeRaw.SetValue(new_exposure_us)
                    print(f"{camera_label}: Exposure updated to {new_exposure_us} us ({new_exposure_us/1000:.2f} ms)")
                    success_messages.append(f"Exposure: {new_exposure_us} µs")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer for exposure!")
                error_occurred = True
            except Exception as e:
                messagebox.showerror("Error", f"Failed to set exposure: {e}")
                error_occurred = True
            
            # Set Gain
            try:
                new_gain = int(gain_entry.get())
                if new_gain < 0:
                    messagebox.showerror("Error", "Gain must be non-negative!")
                    error_occurred = True
                else:
                    camera_capture.camera.GainRaw.SetValue(new_gain)
                    print(f"{camera_label}: Gain updated to {new_gain}")
                    success_messages.append(f"Gain: {new_gain}")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer for gain!")
                error_occurred = True
            except Exception as e:
                messagebox.showerror("Error", f"Failed to set gain: {e}")
                error_occurred = True
            
            # Show success message if no errors occurred
            if not error_occurred and success_messages:
                messagebox.showinfo("Success", f"{camera_label} parameters updated:\n" + "\n".join(success_messages))
        
        set_button = tk.Button(input_window, text="Set Parameters", command=set_parameters, 
                              font=("Arial", 10), bg="lightblue", width=15)
        set_button.pack(pady=10)
        
        # Make window stay on top
        input_window.attributes('-topmost', True)
        
        return input_window
    
    # Create exposure input window
    exposure_window = create_exposure_input_window()
    
    print(f"\n{camera_label} - Live preview started. Press 's' to save, ESC to exit.")
    print(f"{camera_label} - Update rate: {update_rate_hz} Hz ({1/update_rate_hz:.2f} seconds per frame)")
    print(f"{camera_label} - Display scale factor: {scale_factor}")
    print(f"{camera_label} - Max pixel tracking window: {max_buffer_seconds} seconds")
    
    try:
        while True:
            start_time = time.time()
            
            # Update Tkinter window
            try:
                exposure_window.update()
            except tk.TclError:
                # Window was closed
                print(f"{camera_label} - Exposure input window closed. Exiting...")
                break
            
            # Capture image
            img = camera_capture.capture_single_image()
            
            # Get camera parameters and frame properties
            camera_params = camera_capture.get_camera_parameters()
            
            # Add camera label to parameters
            camera_params['Camera Type'] = camera_label
            
            frame_properties = analyze_frame_properties(img)
            
            # Update max pixel buffer with current max pixel value
            current_max_pixel = np.max(img)
            max_pixel_buffer.add_value(current_max_pixel)
            
            # Add windowed maximum to frame properties (normalized to 0-100)
            windowed_max = max_pixel_buffer.get_max()
            if windowed_max is not None:
                buffer_duration = max_pixel_buffer.get_buffer_duration()
                # Normalize to 0-100 scale (assuming 8-bit image with max value 255)
                windowed_max_normalized = (windowed_max / 255.0) * 100.0
                frame_properties[f'{max_buffer_seconds}s Max Pixel'] = f"{windowed_max_normalized:.1f} ({buffer_duration:.1f}s)"
            
            # Add text overlay to image for display
            img_with_info = add_text_to_image(img, camera_params, frame_properties, text_scale)
            
            # Resize for display
            img_display = image_resize(img_with_info, scale_factor)
            
            # Display the image
            cv2.imshow(window_name, img_display)
            
            # Wait for key press based on update rate
            key = cv2.waitKey(wait_time_ms)
            
            if key == 27:  # ESC key to exit
                print(f"{camera_label} - Exiting...")
                break
            elif key == ord('s') or key == ord('S'):  # 's' or 'S' key to save
                # Save the original full-resolution image with info
                filename = save_image_with_info(img, camera_params, frame_properties, save_dir, text_scale)
                print(f"{camera_label} - Image saved: {filename}")
                
                # Print summary
                print(f"\n{camera_label} - Image Properties:")
                for k, v in frame_properties.items():
                    print(f"  {k}: {v}")
                print(f"\n{camera_label} - Camera Parameters:")
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


def run_dual_cameras(camera_0_exposure_us, camera_1_exposure_us, save_dir, text_scale=1.0, 
                     scale_factor=0.6, update_rate_hz=1.0, max_buffer_seconds=10.0):
    """
    Run two cameras simultaneously in separate processes
    
    Args:
        camera_0_exposure_us: int - initial exposure for camera 0
        camera_1_exposure_us: int - initial exposure for camera 1
        save_dir: str - directory to save images
        text_scale: float - text scaling factor for overlays
        scale_factor: float - rescale ratio for display window
        update_rate_hz: float - update rate in Hz (frames per second)
        max_buffer_seconds: float - time window in seconds for max pixel tracking
    """
    # Create two separate processes for each camera
    process_0 = mp.Process(target=camera_process, args=(
        0, camera_0_exposure_us, save_dir, text_scale, scale_factor, 
        update_rate_hz, max_buffer_seconds
    ))
    
    process_1 = mp.Process(target=camera_process, args=(
        1, camera_1_exposure_us, save_dir, text_scale, scale_factor, 
        update_rate_hz, max_buffer_seconds
    ))
    
    # Start both processes
    print("Starting dual camera system...")
    print("="*70)
    process_0.start()
    time.sleep(0.5)  # Small delay to stagger window creation
    process_1.start()
    
    # Wait for both processes to complete
    process_0.join()
    process_1.join()
    
    print("\nDual camera system stopped.")


# DMD constants
DMD_DIM = 1024


if __name__ == "__main__":
    
    # Try to initialize and use the DMD; if it fails, continue in camera-only mode.
    # Note: DMD initialization happens in main process before spawning camera processes
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
    
    print("\n" + "="*70)
    print("DUAL CAMERA SYSTEM")
    print("="*70)
    print("Controls for each camera window:")
    print("  - Press 's' to save image from that camera")
    print("  - Press ESC to exit (closes both cameras)")
    print("  - Use separate exposure control windows for each camera")
    print("="*70 + "\n")
    
    # Configuration
    CAMERA_0_EXPOSURE_US = 150000  # Camera 0 (fiber) initial exposure in microseconds
    CAMERA_0_GAIN = 300              # Camera 0 (fiber) initial gain
    CAMERA_1_EXPOSURE_US = 150000  # Camera 1 (truth) initial exposure in microseconds
    CAMERA_1_GAIN = 0              # Camera 1 (truth) initial gain
    SAVE_DIR = 'C:\\Users\\qiyuanxu\\Desktop\\'  # Directory to save images
    TEXT_SCALE = 1.0           # Text scaling factor for overlays
    SCALE_FACTOR = 0.7         # Display window rescale ratio (0.7 = 70% of original size)
    UPDATE_RATE_HZ = 1.0       # Update rate in Hz (frames per second), default 1.0 Hz = 1 frame/second
    MAX_BUFFER_SECONDS = 10.0  # Time window for max pixel tracking in seconds (default 10.0)
    
    # Start dual camera system
    run_dual_cameras(
        camera_0_exposure_us=CAMERA_0_EXPOSURE_US,
        camera_1_exposure_us=CAMERA_1_EXPOSURE_US,
        save_dir=SAVE_DIR,
        text_scale=TEXT_SCALE,
        scale_factor=SCALE_FACTOR,
        update_rate_hz=UPDATE_RATE_HZ,
        max_buffer_seconds=MAX_BUFFER_SECONDS
    )
