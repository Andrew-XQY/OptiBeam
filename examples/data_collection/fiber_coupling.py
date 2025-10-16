from conf import *
from ALP4 import *
import datetime, time
import cv2
from multiprocessing import Process, Event, Queue
from pypylon import pylon

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
        
        # Set camera to manual mode
        self.camera.ExposureAuto.SetValue('Off')
        self.camera.GainAuto.SetValue('Off')
        self.camera.GammaEnable.SetValue(True)
        
        # Adjust camera settings
        self.camera.ExposureTimeRaw.SetValue(80000)
        self.camera.GainRaw.SetValue(0)
        self.camera.Gamma.SetValue(1.0)
        
        self.print_camera_info()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    def print_camera_info(self):
        info = self.camera.GetDeviceInfo()
        print(''.join(['-']*50))
        print(f"Camera information: {info.GetModelName()}")
        print(f"Camera serial number: {info.GetSerialNumber()}")
        print(''.join(['-']*50))
    
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
                    yield img
                grabResult.Release()
            except Exception as e:
                print("Error encountered: ", e)
                img = cv2.putText(np.zeros((self.output_dim[1], self.output_dim[0]), np.uint8),
                                  "No Image Input", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                yield img
    
    def close(self):
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        self.camera.Close()


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
        img = simulation.macro_pixel(img, size=int(conf['dmd_dim']/img.shape[0]))
        adjusted_img = dmd.dmd_img_adjustment(img, conf['dmd_dim'], angle=conf['dmd_rotation'])
        DMD.display_image(adjusted_img)
        
        # Send the DMD image to the queue for display
        dmd_img_queue.put(adjusted_img)
        time.sleep(1)
    
    DMD.end()


# ============================
# Camera process
# ============================
def camera_process(stop_event, dmd_img_queue, trackbar_queue, conf=None, camera_index=0):
    def on_trackbar(val):
        trackbar_queue.put(val)
    
    cv2.namedWindow("Fiber Coupling")
    cv2.createTrackbar("Special", "Fiber Coupling", 0, 4, on_trackbar)
    
    gen = camera_generator(stop_event, camera_index=camera_index)
    current_dmd_img = None
    
    for frame in gen:
        if stop_event.is_set():
            break
        
        # Check for DMD image updates
        while not dmd_img_queue.empty():
            current_dmd_img = dmd_img_queue.get()
        
        # Crop camera frame if needed
        if conf.get('crop_areas'):
            x1, y1 = conf['crop_areas'][0][0]
            x2, y2 = conf['crop_areas'][0][1]
            frame = frame[y1:y2, x1:x2]
        
        # Create display image with DMD image on the left and camera frame on the right
        if current_dmd_img is not None:
            # Convert DMD image to uint8 if needed
            if current_dmd_img.dtype != np.uint8:
                dmd_display = current_dmd_img.astype(np.uint8)
            else:
                dmd_display = current_dmd_img.copy()
            
            # Resize DMD image to match camera frame height
            dmd_display = cv2.resize(dmd_display, (frame.shape[0], frame.shape[0]))
            
            # Convert to BGR if needed for display
            if len(dmd_display.shape) == 2:
                dmd_display = cv2.cvtColor(dmd_display, cv2.COLOR_GRAY2BGR)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Combine images horizontally
            combined = np.hstack([dmd_display, frame])
        else:
            combined = frame
        
        # Scale for display
        combined = utils.scale_image(combined, 2)
        cv2.imshow("Fiber Coupling", combined)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'esc' to stop
            stop_event.set()
            break
    
    cv2.destroyAllWindows()


# ============================
# Main
# ============================
if __name__ == "__main__":
    # Configuration
    conf = {
        'dmd_dim': 1024,
        'dmd_rotation': 38,
        'dmd_bitDepth': 8,
        'dmd_picture_time': 20000,
        'crop_areas': [((869, 612), (1003, 746))]  # Only one crop area for single camera
    }
    
    # Camera index selection
    CAMERA_INDEX = 0  # Change this to select different camera
    
    # Create stop event and queues
    stop_event = Event()
    dmd_img_queue = Queue()  # For sending DMD images to display
    trackbar_queue = Queue()  # For sending trackbar values to DMD
    
    # Create and start processes
    camera_proc = Process(target=camera_process, args=(stop_event, dmd_img_queue, trackbar_queue, conf, CAMERA_INDEX))
    dmd_proc = Process(target=dmd_process, args=(stop_event, dmd_img_queue, trackbar_queue, conf))
    
    camera_proc.start()
    dmd_proc.start()
    
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping program...")
        stop_event.set()
    
    # Clean termination
    camera_proc.join()
    dmd_proc.join()
