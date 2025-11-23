from conf import *
from pypylon import pylon
from ALP4 import *
import numpy as np
import cv2
import os


class CameraCapture:
    
    def __init__(self, camera_index=0, exposure_ms=40):
        """
        Initialize camera capture
        
        Args:
            camera_index: int - which camera to use (0 for first, 1 for second, etc.)
            exposure_ms: float - exposure time in milliseconds
        """
        self.camera = None
        self.camera_index = camera_index
        self.exposure_ms = exposure_ms
        exposure_us = int(exposure_ms * 1000)
        self.open_camera()
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        
        # Set camera parameters
        self.camera.ExposureAuto.SetValue('Off')
        self.camera.GainAuto.SetValue('Off')
        self.camera.GammaEnable.SetValue(True)
        self.camera.ExposureTimeRaw.SetValue(exposure_us)
        self.camera.GainRaw.SetValue(0)
        self.camera.Gamma.SetValue(1.0)
        
        print(f"Camera {camera_index} initialized")
        
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
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                grabResult.Release()
                return img
            else:
                grabResult.Release()
                raise RuntimeError("Image grab failed")
        finally:
            self.camera.StopGrabbing()
    
    def close(self):
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        self.camera.Close()


def transparency_callback(value):
    """Callback for transparency slider"""
    pass


def overlay_images(camera_img, reference_img, alpha):
    """
    Overlay reference image on camera image with transparency
    
    Args:
        camera_img: np.ndarray - camera frame (grayscale)
        reference_img: np.ndarray - reference image (can be grayscale or color)
        alpha: float - transparency of reference image (0.0 to 1.0)
    
    Returns:
        np.ndarray - merged image
    """
    # Convert camera image to BGR
    if len(camera_img.shape) == 2:
        camera_bgr = cv2.cvtColor(camera_img, cv2.COLOR_GRAY2BGR)
    else:
        camera_bgr = camera_img.copy()
    
    # Convert reference image to BGR if needed
    if len(reference_img.shape) == 2:
        reference_bgr = cv2.cvtColor(reference_img, cv2.COLOR_GRAY2BGR)
    else:
        reference_bgr = reference_img.copy()
    
    # Ensure reference image matches camera image size
    if reference_bgr.shape[:2] != camera_bgr.shape[:2]:
        reference_bgr = cv2.resize(reference_bgr, (camera_bgr.shape[1], camera_bgr.shape[0]))
    
    # Blend images: reference on top with alpha transparency
    merged = cv2.addWeighted(camera_bgr, 1.0, reference_bgr, alpha, 0)
    
    return merged


def align_dmd_live(reference_image_path, save_dir, camera_index=0, exposure_ms=40):
    """
    Display live camera feed with reference image overlay and transparency control
    
    Args:
        reference_image_path: str - path to reference .png image
        save_dir: str - directory to save merged images
        camera_index: int - which camera to use (0 for first, 1 for second, etc.)
        exposure_ms: float - exposure time in milliseconds
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Load reference image
    if not os.path.exists(reference_image_path):
        raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
    
    reference_img = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)
    if reference_img is None:
        raise ValueError(f"Failed to load reference image: {reference_image_path}")
    
    print(f"Reference image loaded: {reference_image_path}")
    print(f"Reference image size: {reference_img.shape}")
    
    # Initialize camera
    camera_capture = CameraCapture(camera_index=camera_index, exposure_ms=exposure_ms)
    
    # Create CV2 window
    window_name = 'DMD Alignment'
    cv2.namedWindow(window_name)
    
    # Create transparency slider (0-100, default 50 for half transparency)
    cv2.createTrackbar('Transparency', window_name, 0, 100, transparency_callback)
    
    print("\nControls:")
    print("  - Adjust 'Transparency' slider to control overlay opacity")
    print("  - Press 's' to save the merged image")
    print("  - Press ESC to exit")
    
    try:
        while True:
            # Capture camera image
            camera_img = camera_capture.capture_single_image()
            
            # Get transparency value from slider (convert 0-100 to 0.0-1.0)
            transparency_value = cv2.getTrackbarPos('Transparency', window_name)
            alpha = transparency_value / 100.0
            
            # Overlay images
            merged_img = overlay_images(camera_img, reference_img, alpha)
            
            # Display the merged image
            cv2.imshow(window_name, merged_img)
            
            # Wait for key press
            key = cv2.waitKey(30)
            
            if key == 27:  # ESC key to exit
                print("Exiting...")
                break
            elif key == ord('s') or key == ord('S'):  # 's' or 'S' key to save
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(save_dir, f"aligned_{timestamp}.png")
                cv2.imwrite(filename, merged_img)
                print(f"Merged image saved: {filename}")
                
    finally:
        camera_capture.close()
        cv2.destroyAllWindows()


# DMD constants
DMD_DIM = 1024


if __name__ == "__main__":
    # Configuration
    REFERENCE_IMAGE_PATH = 'C:\\Users\\qiyuanxu\\Documents\\GitHub\\OptiBeam\\examples\\data_collection\\results\\20251118_110017.png'  # Path to reference image
    SAVE_DIR = 'C:\\Users\\qiyuanxu\\Desktop\\'  # Directory to save merged images
    CAMERA_INDEX = 1  # Which camera to use (0 for first, 1 for second, etc.)
    EXPOSURE_MS = 200  # Exposure time in milliseconds
    
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
    
    # Start alignment tool
    align_dmd_live(
        reference_image_path=REFERENCE_IMAGE_PATH,
        save_dir=SAVE_DIR,
        camera_index=CAMERA_INDEX,
        exposure_ms=EXPOSURE_MS
    )
