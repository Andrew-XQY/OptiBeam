from conf import *
from ALP4 import *
import datetime, time
import cv2
from multiprocessing import Process, Event, Queue
from pypylon import pylon
import numpy as np
import csv
import os

# ----------------------------
# Helpers (analysis and display)
# ----------------------------
def get_camera_parameters(camera):
    """
    Get camera core parameters that affect the image
    
    Args:
        camera: pylon.InstantCamera
        
    Returns:
        dict: camera parameters
    """
    params = {}
    try:
        if hasattr(camera, "ExposureTimeRaw"):
            params['Exposure (ms)'] = f'{camera.ExposureTimeRaw.Value / 1000:.2f}'
        elif hasattr(camera, "ExposureTime"):
            params['Exposure (ms)'] = f'{camera.ExposureTime.Value / 1000:.2f}'
        
        if hasattr(camera, "GainRaw"):
            params['Gain'] = f'{camera.GainRaw.Value}'
        elif hasattr(camera, "Gain"):
            params['Gain'] = f'{camera.Gain.Value:.2f}'
        
        if hasattr(camera, "Gamma"):
            params['Gamma'] = f'{camera.Gamma.Value:.2f}'
    except Exception as e:
        print(f"Error getting camera parameters: {e}")
    
    return params


def analyze_frame_properties(image, normalize_range=(0, 100), peak_buffer=None):
    """
    Analyze frame properties and normalize them to a specified range
    
    Args:
        image: np.ndarray
        normalize_range: tuple, default (0, 100)
        peak_buffer: dict, optional buffer to track maximum values ever reached
    
    Returns:
        dict: frame properties with normalized values and raw values
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
        'Max Pixel': f'{normalized_max:.2f}',
        'Total Sum': f'{normalized_sum:.2f}',
        'raw_max_pixel': max_pixel,  # Add raw values
        'raw_total_sum': total_sum
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
        properties['Peak Max'] = f'{peak_buffer["peak_max_pixel"]:.2f}'
        properties['Peak Sum'] = f'{peak_buffer["peak_total_sum"]:.2f}'
    
    return properties


def scale_for_display(img, scale=None, max_side=1200):
    """
    Uniformly resize the combined image for preview only (keeps aspect).
    If scale is None, fit the longest side to max_side (<= max_side).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    if scale is not None and scale > 0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
    else:
        f = min(max_side / float(max(h, w)), 1.0)
        new_w = max(1, int(round(w * f)))
        new_h = max(1, int(round(h * f)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def prepare_dmd_display(dmd_img, target_height):
    """
    Convert DMD image to uint8 (0..255) and resize *proportionally* so its height
    equals target_height while preserving the DMD's native aspect ratio.
    """
    # Normalize to visible 8-bit
    if dmd_img.dtype != np.uint8:
        arr = dmd_img.astype(np.float32)
        mx = float(arr.max())
        if mx > 0:
            dmd_display = np.clip(255.0 * (arr / mx), 0, 255).astype(np.uint8)
        else:
            dmd_display = np.zeros_like(dmd_img, dtype=np.uint8)
    else:
        dmd_display = dmd_img.copy()

    # Ensure 3-channel BGR for stacking with camera BGR
    if dmd_display.ndim == 2:
        dmd_display = cv2.cvtColor(dmd_display, cv2.COLOR_GRAY2BGR)

    dh, dw = dmd_display.shape[:2]
    if dh <= 0 or dw <= 0 or target_height <= 0:
        return dmd_display

    scale = target_height / float(dh)
    new_w = max(1, int(round(dw * scale)))
    return cv2.resize(dmd_display, (new_w, target_height), interpolation=cv2.INTER_AREA)

# ============================
# Single Camera Manager
# ============================
class SingleCameraCapture:
    def __init__(self, camera_index=0):
        self.camera = None
        self.camera_index = camera_index
        self.open_camera()
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.output_dim = [self.camera.Width.GetValue(), self.camera.Height.GetValue()]

        # Manual exposure/gain
        self.camera.ExposureAuto.SetValue('Off')
        self.camera.GainAuto.SetValue('Off')
        if hasattr(self.camera, "GammaEnable"):
            self.camera.GammaEnable.SetValue(False)

        # Adjust settings (adapt names if your model differs)
        if hasattr(self.camera, "ExposureTimeRaw"):
            self.camera.ExposureTimeRaw.SetValue(80000)
        else:
            self.camera.ExposureTime.SetValue(80000.0)
        if hasattr(self.camera, "GainRaw"):
            self.camera.GainRaw.SetValue(200)
        else:
            self.camera.Gain.SetValue(200.0)
        if hasattr(self.camera, "Gamma"):
            self.camera.Gamma.SetValue(1.0)

        self.print_camera_info()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def print_camera_info(self):
        info = self.camera.GetDeviceInfo()
        print('-' * 50)
        print(f"Camera information: {info.GetModelName()}")
        print(f"Camera serial number: {info.GetSerialNumber()}")
        print('-' * 50)

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
        while True:
            if not self.camera.IsGrabbing():
                self.open_camera()
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            try:
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    image = self.converter.Convert(grabResult)
                    img = image.GetArray()
                    yield np.ascontiguousarray(img)
                grabResult.Release()
            except Exception as e:
                print("Error encountered: ", e)
                img = cv2.putText(
                    np.zeros((self.output_dim[1], self.output_dim[0]), np.uint8),
                    "No Image Input", (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2
                )
                yield img

    def close(self):
        try:
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            if self.camera:
                self.camera.Close()
        except Exception:
            pass

# ============================
# Camera generator
# ============================
def camera_generator(stop_event, camera_index=0):
    cam_capture = SingleCameraCapture(camera_index=camera_index)
    try:
        for frame in cam_capture.capture():
            if stop_event.is_set():
                break
            yield frame, cam_capture.camera  # Also return camera object for parameter access
    finally:
        cam_capture.close()

# ============================
# DMD process
# ============================
def dmd_process(stop_event, dmd_img_queue, trackbar_queue, dmd_state_queue, conf=None):
    DMD = dmd.ViALUXDMD(ALP4(version='4.3'))
    calibrator = simulation.CornerBlocksCalibrator(block_size=16)

    while not stop_event.is_set():
        if not trackbar_queue.empty():
            calibrator.set_special(trackbar_queue.get())
        calibrator.generate_blocks()
        img = calibrator.canvas
        img = simulation.macro_pixel(img, size=int(conf['dmd_dim'] / img.shape[0]))
        adjusted_img = dmd.dmd_img_adjustment(img, conf['dmd_dim'], angle=DMD_ROTATION_ANGLE)
        DMD.display_image(adjusted_img)

        # Non-blocking handoff (drop backlog)
        try:
            while dmd_img_queue.qsize() > 1:
                _ = dmd_img_queue.get_nowait()
        except Exception:
            pass
        dmd_img_queue.put(img)  # Send the image before adjustment for display
        
        # Share the current state_index
        try:
            while dmd_state_queue.qsize() > 1:
                _ = dmd_state_queue.get_nowait()
        except Exception:
            pass
        dmd_state_queue.put((calibrator.special, calibrator.state_index))

        time.sleep(0.5)

    DMD.end()

# ============================
# Camera process
# ============================
def camera_process(stop_event, dmd_img_queue, trackbar_queue, dmd_state_queue, conf=None, camera_index=0, display_scale=None, text_scale=0.5):
    def on_trackbar(val):
        trackbar_queue.put(val)
    
    def on_exposure_trackbar(val):
        # Set exposure time in microseconds (trackbar value is in milliseconds)
        if camera_obj is not None:
            try:
                if hasattr(camera_obj, "ExposureTimeRaw"):
                    camera_obj.ExposureTimeRaw.SetValue(val * 1000)
                elif hasattr(camera_obj, "ExposureTime"):
                    camera_obj.ExposureTime.SetValue(val * 1000.0)
            except Exception as e:
                print(f"Error setting exposure: {e}")
    
    def on_gain_trackbar(val):
        # Set gain value
        if camera_obj is not None:
            try:
                if hasattr(camera_obj, "GainRaw"):
                    camera_obj.GainRaw.SetValue(val)
                elif hasattr(camera_obj, "Gain"):
                    camera_obj.Gain.SetValue(float(val))
            except Exception as e:
                print(f"Error setting gain: {e}")

    # Keep window aspect ratio, allow resizing
    cv2.namedWindow("Fiber Coupling", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    try:
        cv2.setWindowProperty("Fiber Coupling", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    except Exception:
        # Older OpenCV may not support setting this property; WINDOW_KEEPRATIO flag is enough.
        pass
    cv2.resizeWindow("Fiber Coupling", 800, 600)
    
    # Create trackbars
    cv2.createTrackbar("Pattern", "Fiber Coupling", 0, 4, on_trackbar)
    cv2.createTrackbar("Exposure(ms)", "Fiber Coupling", 80, 500, on_exposure_trackbar)  # 0-500ms
    cv2.createTrackbar("Gain", "Fiber Coupling", 200, 300, on_gain_trackbar)  # 0-300 gain

    gen = camera_generator(stop_event, camera_index=camera_index)
    current_dmd_img = None
    camera_obj = None
    
    # Peak tracking
    peak_buffer = {}
    frame_count = 0
    warmup_frames = 50  # Skip first 50 frames before tracking peaks
    
    # Data recording list
    frame_data_list = []
    
    # DMD state tracking
    current_dmd_special = 0
    current_dmd_state_index = 0
    
    # Text display parameters (controlled by text_scale)
    thickness = max(1, int(text_scale * 2))
    line_spacing = int(text_scale * 40)

    for frame, camera_obj in gen:
        if stop_event.is_set():
            break
        
        frame_count += 1

        # Pull latest DMD image (drain queue)
        while not dmd_img_queue.empty():
            current_dmd_img = dmd_img_queue.get()
        
        # Pull latest DMD state (drain queue)
        while not dmd_state_queue.empty():
            current_dmd_special, current_dmd_state_index = dmd_state_queue.get()

        # Base: camera frame (keep its aspect unchanged)
        cam_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
        
        # Analyze frame properties (use original grayscale frame for analysis - ONLY camera frame)
        properties = analyze_frame_properties(
            frame, 
            peak_buffer=peak_buffer if frame_count > warmup_frames else None
        )
        
        # Get camera parameters
        camera_params = get_camera_parameters(camera_obj) if camera_obj else {}
        
        # Record data after warmup frames
        if frame_count >= warmup_frames and camera_obj is not None:
            try:
                # Get camera exposure
                if hasattr(camera_obj, "ExposureTimeRaw"):
                    exposure = camera_obj.ExposureTimeRaw.Value / 1000  # Convert to ms
                elif hasattr(camera_obj, "ExposureTime"):
                    exposure = camera_obj.ExposureTime.Value / 1000  # Convert to ms
                else:
                    exposure = 0
                
                # Get camera gain
                if hasattr(camera_obj, "GainRaw"):
                    gain = camera_obj.GainRaw.Value
                elif hasattr(camera_obj, "Gain"):
                    gain = camera_obj.Gain.Value
                else:
                    gain = 0
                
                # Get block position index from DMD calibrator state
                block_position = current_dmd_state_index
                
                # Get max intensity and total intensity from analyze_frame_properties (use raw values)
                max_intensity = properties['raw_max_pixel']
                total_intensity = properties['raw_total_sum']
                
                # Store as tuple: (exposure, gain, block_position, max_intensity, total_intensity)
                frame_data_list.append((exposure, gain, block_position, max_intensity, total_intensity))
            except Exception as e:
                print(f"Error recording frame data: {e}")

        # Left panel: DMD scaled *proportionally* to camera height
        if current_dmd_img is not None:
            target_h = cam_bgr.shape[0]
            dmd_display = prepare_dmd_display(current_dmd_img, target_height=target_h)
            # Create a green separator line
            separator = np.zeros((target_h, 3, 3), dtype=np.uint8)
            separator[:] = (0, 255, 0)  # Green color in BGR
            combined = np.hstack([dmd_display, separator, cam_bgr])
        else:
            combined = cam_bgr

        # Optional uniform downscale for preview (does not distort camera ratio)
        preview = scale_for_display(combined, scale=display_scale, max_side=1200)
        
        # Create status bar at the bottom with text information
        status_height = max(80, int(100 * text_scale))  # Height of status bar
        status_bar = np.zeros((status_height, preview.shape[1], 3), dtype=np.uint8)
        status_bar[:] = (40, 40, 40)  # Dark gray background
        
        # Combine all info into one line or two lines
        # Line 1: Frame properties (white text, left side)
        info_texts = []
        for key, value in properties.items():
            # Skip raw values from display (they're only for CSV recording)
            if key not in ['raw_max_pixel', 'raw_total_sum']:
                info_texts.append(f'{key}: {value}')
        
        # Line 2: Camera parameters (blue text, left side)
        camera_texts = []
        for key, value in camera_params.items():
            camera_texts.append(f'{key}: {value}')
        
        # Draw frame properties on first line
        properties_line = '  |  '.join(info_texts)
        y_pos = int(30 * text_scale)
        cv2.putText(status_bar, properties_line, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.8, (255, 255, 255), thickness)
        
        # Draw camera parameters on second line
        camera_line = '  |  '.join(camera_texts)
        y_pos = int(30 * text_scale) + line_spacing
        cv2.putText(status_bar, camera_line, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.8, (100, 150, 255), thickness)
        
        # Stack the preview image with status bar
        preview = np.vstack([preview, status_bar])

        # Match window size to image size (prevents GUI squeezing)
        h, w = preview.shape[:2]
        cv2.resizeWindow("Fiber Coupling", int(w), int(h))
        cv2.imshow("Fiber Coupling", preview)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
            stop_event.set()
            break

    # Save recorded data to CSV
    if frame_data_list and conf.get('csv_save_dir'):
        try:
            save_dir = conf['csv_save_dir']
            if save_dir and os.path.exists(save_dir):
                # Generate timestamp filename
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                csv_filename = f"{timestamp}.csv"
                csv_path = os.path.join(save_dir, csv_filename)
                
                # Write to CSV with header
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['camera_exposure_ms', 'camera_gain', 'block_position_index', 'max_intensity', 'total_intensity'])
                    writer.writerows(frame_data_list)
                
                print(f"Data saved to: {csv_path}")
            else:
                print(f"CSV save directory does not exist or is not configured: {save_dir}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")

    cv2.destroyAllWindows()

# ============================
# Main
# ============================
if __name__ == "__main__":
    conf = {
        'dmd_dim': 1024,
        'dmd_rotation': 0,   # DMD rotation angle for image orientation correction
        'dmd_bitDepth': 8,
        'dmd_picture_time': 20000,
        'csv_save_dir': 'C:\\Users\\qiyuanxu\\Desktop\\fiber_coupling_experiments_2025\\' 
    }

    # Use None to auto-fit, or a fraction like 0.5. Avoid >1 with full frames.
    DISPLAY_SCALE = 0.45
    CAMERA_INDEX = 0
    TEXT_SCALE = 1  # Control text size: 0.3=small, 0.5=medium, 0.8=large, 1.0=very large

    stop_event = Event()
    dmd_img_queue = Queue()
    trackbar_queue = Queue()
    dmd_state_queue = Queue()  # New queue for DMD state sharing

    camera_proc = Process(
        target=camera_process,
        args=(stop_event, dmd_img_queue, trackbar_queue, dmd_state_queue, conf, CAMERA_INDEX, DISPLAY_SCALE, TEXT_SCALE)
    )
    dmd_proc = Process(
        target=dmd_process,
        args=(stop_event, dmd_img_queue, trackbar_queue, dmd_state_queue, conf)
    )

    camera_proc.start()
    dmd_proc.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping program...")
        stop_event.set()

    camera_proc.join()
    dmd_proc.join()
