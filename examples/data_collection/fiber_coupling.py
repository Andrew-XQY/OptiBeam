from conf import *
from ALP4 import *
import datetime, time
import cv2
from multiprocessing import Process, Event, Queue
from pypylon import pylon
import numpy as np

# ----------------------------
# Helpers (display-safe resizes)
# ----------------------------
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
            self.camera.GammaEnable.SetValue(True)

        # Adjust settings (adapt names if your model differs)
        if hasattr(self.camera, "ExposureTimeRaw"):
            self.camera.ExposureTimeRaw.SetValue(80000)
        else:
            self.camera.ExposureTime.SetValue(80000.0)
        if hasattr(self.camera, "GainRaw"):
            self.camera.GainRaw.SetValue(0)
        else:
            self.camera.Gain.SetValue(0.0)
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
            yield frame
    finally:
        cam_capture.close()

# ============================
# DMD process
# ============================
def dmd_process(stop_event, dmd_img_queue, trackbar_queue, conf=None):
    DMD = dmd.ViALUXDMD(ALP4(version='4.3'))
    calibrator = simulation.CornerBlocksCalibrator(block_size=32)

    while not stop_event.is_set():
        if not trackbar_queue.empty():
            calibrator.set_special(trackbar_queue.get())
        calibrator.generate_blocks()
        img = calibrator.canvas
        img = simulation.macro_pixel(img, size=int(conf['dmd_dim'] / img.shape[0]))
        adjusted_img = dmd.dmd_img_adjustment(img, conf['dmd_dim'], angle=conf['dmd_rotation'])
        DMD.display_image(adjusted_img)

        # Non-blocking handoff (drop backlog)
        try:
            while dmd_img_queue.qsize() > 1:
                _ = dmd_img_queue.get_nowait()
        except Exception:
            pass
        dmd_img_queue.put(adjusted_img)

        time.sleep(0.5)

    DMD.end()

# ============================
# Camera process
# ============================
def camera_process(stop_event, dmd_img_queue, trackbar_queue, conf=None, camera_index=0, display_scale=None):
    def on_trackbar(val):
        trackbar_queue.put(val)

    # Keep window aspect ratio, allow resizing
    cv2.namedWindow("Fiber Coupling", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    try:
        cv2.setWindowProperty("Fiber Coupling", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    except Exception:
        # Older OpenCV may not support setting this property; WINDOW_KEEPRATIO flag is enough.
        pass
    cv2.resizeWindow("Fiber Coupling", 800, 600)
    cv2.createTrackbar("Special", "Fiber Coupling", 0, 4, on_trackbar)

    gen = camera_generator(stop_event, camera_index=camera_index)
    current_dmd_img = None

    for frame in gen:
        if stop_event.is_set():
            break

        # Pull latest DMD image (drain queue)
        while not dmd_img_queue.empty():
            current_dmd_img = dmd_img_queue.get()

        # Base: camera frame (keep its aspect unchanged)
        cam_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame

        # Left panel: DMD scaled *proportionally* to camera height
        if current_dmd_img is not None:
            target_h = cam_bgr.shape[0]
            dmd_display = prepare_dmd_display(current_dmd_img, target_height=target_h)
            combined = np.hstack([dmd_display, cam_bgr])
        else:
            combined = cam_bgr

        # Optional uniform downscale for preview (does not distort camera ratio)
        preview = scale_for_display(combined, scale=display_scale, max_side=1200)

        # Match window size to image size (prevents GUI squeezing)
        h, w = preview.shape[:2]
        cv2.resizeWindow("Fiber Coupling", int(w), int(h))
        cv2.imshow("Fiber Coupling", preview)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
            stop_event.set()
            break

    cv2.destroyAllWindows()

# ============================
# Main
# ============================
if __name__ == "__main__":
    conf = {
        'dmd_dim': 1024,
        'dmd_rotation': 0,   # DMD rotation angle for image orientation correction
        'dmd_bitDepth': 8,
        'dmd_picture_time': 20000
    }

    # Use None to auto-fit, or a fraction like 0.5. Avoid >1 with full frames.
    DISPLAY_SCALE = None
    CAMERA_INDEX = 0

    stop_event = Event()
    dmd_img_queue = Queue()
    trackbar_queue = Queue()

    camera_proc = Process(
        target=camera_process,
        args=(stop_event, dmd_img_queue, trackbar_queue, conf, CAMERA_INDEX, DISPLAY_SCALE)
    )
    dmd_proc = Process(
        target=dmd_process,
        args=(stop_event, dmd_img_queue, trackbar_queue, conf)
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
