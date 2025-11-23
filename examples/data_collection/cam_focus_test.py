from conf import *
from pypylon import pylon
from datetime import datetime

import cv2
import os
import numpy as np


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
    
    
    def __init__(self, camera_index=0, initial_exposure_ms=40):
        self.camera = None
        self.camera_index = camera_index
        self.open_camera()
        self.converter = pylon.ImageFormatConverter()
        # Setting the converter to output mono8 images for simplicity
        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.output_dim = [self.camera.Width.GetValue(), self.camera.Height.GetValue()]
        
        # Ensure the camera exposure, gain, and gamma are set to manual mode before adjusting
        self.camera.ExposureAuto.SetValue('Off')  # Turn off auto exposure
        self.camera.GainAuto.SetValue('Off')      # Turn off auto gain
        self.camera.GammaEnable.SetValue(True)    # Enable gamma correction if supported
        
        # Adjust camera settings - these values are examples and should be adjusted based on your needs and camera capabilities
        self.camera.ExposureTimeRaw.SetValue(initial_exposure_ms * 1000)  # Convert milliseconds to microseconds
        self.camera.GainRaw.SetValue(0)            # Set gain
        self.camera.Gamma.SetValue(1.0)              # Set gamma value to 1.0 (if supported)
        self.print_camera_info()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
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
    
    def capture(self):
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

    def close(self):
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        self.camera.Close()



def image_resize(img, scale_percent):
    # Calculate the new dimensions based on a scaling factor (e.g., 50% of the original size)
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv2.resize(img, dim) 



def calculate_image_sharpness(image):
    # Compute gradients along the x and y axis, respectively
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # Compute the square root of the sum of the squared gradients
    focus_measure = np.sqrt(gx**2 + gy**2).mean()
    return focus_measure


def get_camera_parameters(camera):
    """
    Get camera core parameters that affect the image
    
    Args:
        camera: pylon.InstantCamera
        
    Returns:
        dict: camera parameters
    """
    return {
        'Exposure (ms)': f'{camera.ExposureTimeRaw.Value / 1000:.2f}',
        'Gain': f'{camera.GainRaw.Value}',
        'Gamma': f'{camera.Gamma.Value:.2f}'
    }


def analyze_frame_properties(image, normalize_range=(0, 100), peak_buffer=None):
    """
    Analyze frame properties and normalize them to a specified range
    
    Args:
        image: np.ndarray
        normalize_range: tuple, default (0, 100)
        peak_buffer: dict, optional buffer to track maximum values ever reached
    
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
    
    # Track peak values if buffer is provided
    if peak_buffer is not None:
        # Update peak max pixel value
        if 'peak_max_pixel' not in peak_buffer or normalized_max > peak_buffer['peak_max_pixel']:
            peak_buffer['peak_max_pixel'] = normalized_max
        
        # Update peak total sum
        if 'peak_total_sum' not in peak_buffer or normalized_sum > peak_buffer['peak_total_sum']:
            peak_buffer['peak_total_sum'] = normalized_sum
        
        # Add peak values to properties
        properties['Peak Max Pixel'] = f'{peak_buffer["peak_max_pixel"]:.2f}'
        properties['Peak Total Sum'] = f'{peak_buffer["peak_total_sum"]:.2f}'
    
    return properties


# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global click_position 
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_position is None:
            # First click: enable magnify mode
            click_position = (x, y)
        else:
            # Second click: reset to original view
            click_position = None


def draw_crosshair_with_ticks(image, major_tick_spacing=50, color=(0, 255, 0), thickness=1):
    """
    Draw a crosshair with major and minor ticks on the image
    
    Args:
        image: np.ndarray - input image
        major_tick_spacing: int - pixel spacing between major ticks
        color: tuple - BGR color for the crosshair and ticks
        thickness: int - line thickness
    
    Returns:
        np.ndarray - image with crosshair drawn
    """
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    
    # Calculate minor tick spacing (5 subdivisions between major ticks)
    minor_tick_spacing = major_tick_spacing // 5
    
    # Define tick lengths
    major_tick_length = 15
    minor_tick_length = 8
    
    # Draw vertical center line
    cv2.line(image, (center_x, 0), (center_x, height), color, thickness)
    
    # Draw horizontal center line
    cv2.line(image, (0, center_y), (width, center_y), color, thickness)
    
    # Draw vertical ticks along the horizontal center line
    # Calculate how many ticks fit on each side of center
    num_ticks_left = center_x // minor_tick_spacing
    num_ticks_right = (width - center_x) // minor_tick_spacing
    
    for i in range(-num_ticks_left, num_ticks_right + 1):
        x = center_x + i * minor_tick_spacing
        if 0 <= x < width:
            # Check if this is a major tick (at major_tick_spacing intervals from center)
            if i % (major_tick_spacing // minor_tick_spacing) == 0:
                tick_length = major_tick_length
            else:
                tick_length = minor_tick_length
            
            # Draw tick extending from center line
            cv2.line(image, (x, center_y - tick_length // 2), 
                    (x, center_y + tick_length // 2), color, thickness)
    
    # Draw horizontal ticks along the vertical center line
    # Calculate how many ticks fit on each side of center
    num_ticks_top = center_y // minor_tick_spacing
    num_ticks_bottom = (height - center_y) // minor_tick_spacing
    
    for i in range(-num_ticks_top, num_ticks_bottom + 1):
        y = center_y + i * minor_tick_spacing
        if 0 <= y < height:
            # Check if this is a major tick (at major_tick_spacing intervals from center)
            if i % (major_tick_spacing // minor_tick_spacing) == 0:
                tick_length = major_tick_length
            else:
                tick_length = minor_tick_length
            
            # Draw tick extending from center line
            cv2.line(image, (center_x - tick_length // 2, y), 
                    (center_x + tick_length // 2, y), color, thickness)
    
    return image



# Display the image with the red box and magnified area
def display_image(save_to='', scale_factor=0.5, intensity_monitor=False, camera_index=0, text_scale=1.0, initial_exposure_ms=40):
    
    coupling = processing.IntensityMonitor()
    
    parent_directory = os.path.dirname(os.path.realpath(__file__))
    camera_capture = CameraCapture(camera_index=camera_index, initial_exposure_ms=initial_exposure_ms)
    cv2.namedWindow('Camera Output')
    cv2.setMouseCallback('Camera Output', mouse_callback)
    
    cv2.createTrackbar('Exposure time (ms)', 'Camera Output', initial_exposure_ms, 500, 
                       lambda x: camera_capture.camera.ExposureTimeRaw.SetValue(x*1000))  # miniseconds
    cv2.createTrackbar('Gain', 'Camera Output', 0, 300, 
                       lambda x: camera_capture.camera.GainRaw.SetValue(x))
    
    # Initialize peak buffer to track maximum values
    peak_buffer = {}
    frame_count = 0
    warmup_frames = 50  # Skip first 100 frames before tracking peaks
    last_img = None  # Store the last captured image for saving on ESC
    
    try:
        for img in camera_capture.capture():
            last_img = img  # Keep track of the last captured image
            resized_img = image_resize(img, scale_factor)  # only for display, not for saving
            frame_count += 1
            # Only pass peak_buffer after warmup period
            properties = analyze_frame_properties(resized_img, peak_buffer=peak_buffer if frame_count > warmup_frames else None)
            if click_position:
                # Parameters for the red box and magnified area
                box_size = 170  # Size of the red box
                magnification = 4  # Magnification factor for the cropped area
                #click_position = (int(click_position[0] * (1+scale_factor)), int(click_position[1] * (1+scale_factor)))
                
                # Calculate the top left and bottom right points for the red box
                x1 = max(click_position[0] - box_size // 2, 0)
                y1 = max(click_position[1] - box_size // 2, 0)
                x2 = min(click_position[0] + box_size // 2, resized_img.shape[1] - 1)
                y2 = min(click_position[1] + box_size // 2, resized_img.shape[0] - 1)
                
                # Crop and magnify the area within the red box
                cropped_img = resized_img[y1:y2, x1:x2]
                magnified_img = cv2.resize(cropped_img, (0, 0), fx=magnification, fy=magnification)
                
                # Overlay the magnified image on the top left corner of the original image
                resized_img[0:magnified_img.shape[0], 0:magnified_img.shape[1]] = magnified_img
                
                properties = analyze_frame_properties(magnified_img, peak_buffer=peak_buffer if frame_count > warmup_frames else None)
            
            # Display the properties on the image (top left, white)
            y_position = int(30 * text_scale)
            line_spacing = int(35 * text_scale)
            thickness = max(1, int(2 * text_scale))
            for key, value in properties.items():
                text = f'{key}: {value}'
                cv2.putText(resized_img, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), thickness)
                y_position += line_spacing
            
            # Display camera parameters on the image (bottom right, blue)
            camera_params = get_camera_parameters(camera_capture.camera)
            y_position = resized_img.shape[0] - int(20 * text_scale)  # Start from bottom
            for key, value in reversed(list(camera_params.items())):
                text = f'{key}: {value}'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)[0]
                x_position = resized_img.shape[1] - text_size[0] - 10  # Right aligned
                cv2.putText(resized_img, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 0, 0), thickness)
                y_position -= line_spacing
            
            if intensity_monitor:
                    intensity = coupling.add_image(resized_img)
                    resized_img = utils.join_images([resized_img,intensity])
            
            # Convert grayscale to BGR so we can draw colored crosshair
            if len(resized_img.shape) == 2:  # If grayscale
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
            
            # Add crosshair with major and minor ticks
            resized_img = draw_crosshair_with_ticks(resized_img, major_tick_spacing=50, 
                                                    color=(0, 255, 0), thickness=1)
            
            cv2.imshow('Camera Output', resized_img)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC key to exit
                # Save the last image before exiting
                if last_img is not None:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{parent_directory}/{save_to}/{timestamp}.png"
                    cv2.imwrite(filename, last_img)
                    print(f"Last image saved as {filename}")
                break
            elif key == ord('s'):  # 's' key to save the image
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{parent_directory}/{save_to}/{timestamp}.png"
                cv2.imwrite(filename, img)
                print(f"Image saved as {filename}")
    finally:
        camera_capture.close()
        cv2.destroyAllWindows()


from ALP4 import *
DMD_DIM = 1024

# Using real image load on the DMD for testing
# def read_gray_square(path):
#     import numpy as np
#     from PIL import Image
#     """Read image, return single-channel square numpy array."""
#     img = Image.open(path).convert('L')
#     s = min(img.size)  # no upscaling
#     return np.array(img.resize((s, s), Image.LANCZOS))

# calibration_img = read_gray_square("C:\\Users\\qiyuanxu\\Downloads\\gradient.png")


if __name__ == "__main__":
    # Camera settings
    INITIAL_EXPOSURE_MS = 195 #nitial camera exposure in milliseconds
    
    save_dir = 'C:\\Users\\qiyuanxu\\Desktop\\'
    # Try to initialize and use the DMD; if it fails, continue in camera-only mode.
    try:
        DMD = dmd.ViALUXDMD(ALP4(version='4.3'))
        # calibration_img = simulation.generate_radial_gradient(size=DMD_DIM)
        # calibration_img = np.ones((256, 256)) * 255  # 0-255 grayscale?
        calibration_img = simulation.generate_inverted_upward_arrow(intesity=255)
        
        # calibration_img = simulation.dmd_calibration_pattern_generation()3
        
        calibration_img = simulation.macro_pixel(calibration_img, size=int(DMD_DIM/calibration_img.shape[0])) 
        calibration_img = dmd.dmd_img_adjustment(calibration_img, DMD_DIM, angle=DMD_ROTATION_ANGLE)

        DMD.display_image(calibration_img)  # preload one image for camera calibration
    except Exception as e:
        print(f"Warning: DMD initialization or display failed ({e}). Running in camera-only mode.")

    click_position = None
    # Use camera_index=0 for first camera, camera_index=1 for second camera
    display_image('results', camera_index=1, text_scale=1, scale_factor=0.7, initial_exposure_ms=INITIAL_EXPOSURE_MS)
