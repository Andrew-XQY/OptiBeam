from pypylon import pylon
import numpy as np
import cv2
import time
import warnings


framerate = 30.
exposure = 5.
gain = 1.
compression_ratio = 70.
pixel_scaling = 2.79
num_frames1 = 0
num_frames2 = 0

# Create VideoWriters for both cameras
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('D:\\videoCaptures\\output1.avi', fourcc, framerate, (1920,1200), isColor=False)
out2 = cv2.VideoWriter('D:\\videoCaptures\\output2.avi', fourcc, framerate, (1920,1200), isColor=False)

def init_ptp(camera, index):
    # PTP setup
    camera.PtpEnable.SetValue(False)
    camera.BslPtpPriority1.SetValue(128)
    camera.BslPtpProfile.SetValue("DelayRequestResponseDefaultProfile")
    camera.BslPtpNetworkMode.SetValue("Multicast")
    camera.BslPtpManagementEnable.SetValue(False)
    camera.BslPtpTwoStep.SetValue(False)
    camera.PtpEnable.SetValue(True)

def check_ptp(cameras):
    initialized = False
    locked = False

    # Wait until correctly initialized or timeout
    time1 = time.time()
    while not initialized:
        status_arr = np.zeros(cameras.GetSize(), dtype=np.bool_)
        for i, camera in enumerate(cameras):
            camera.PtpDataSetLatch.Execute()
            status_arr[i] = (camera.PtpStatus.GetValue() == 'Master' \
                            or camera.PtpStatus.GetValue() != 'Initializing')
        initialized = np.all(status_arr)
        if (time.time() - time1) > 3:
            if not initialized:
                warnings.warn('PTP not initialized -> Timeout')
            break

    # If correctly initialized, wait until settled or timeout
    if initialized:
        time2 = time.time()
        while not locked:
            status_arr = np.zeros(cameras.GetSize(), dtype=np.bool_)
            status_string = ''
            for i, camera in enumerate(cameras):
                camera.PtpDataSetLatch.Execute()
                status_arr[i] = (camera.PtpStatus.GetValue() == 'Master' \
                                or camera.PtpServoStatus.GetValue() == 'Locked')
                status_string += 'Camera {:d} locked: {} | '.format(i, status_arr[i])
            print(status_string)
            locked = np.all(status_arr)
            if (time.time() - time2) > 30:
                if not locked:
                    warnings.warn('PTP not locked -> Timeout')
                break

    return initialized and locked

def init_camera(camera, index):
    camera.GainAuto.SetValue("Off")
    camera.Gain.SetValue(gain)

    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTime.SetValue(int(exposure*1000))

    # Beyond Pixel setup
    camera.PixelFormat.SetValue("Mono8")
    # camera.BslScalingFactor.SetValue(pixel_scaling)

    # Beyond Compression setup
    # camera.ImageCompressionMode.SetValue("BaslerCompressionBeyond")
    # camera.ImageCompressionRateOption.SetValue("FixRatio")
    # camera.BslImageCompressionRatio.SetValue(compression_ratio)

    # Periodic Signal setup
    if camera.BslPeriodicSignalSource.GetValue() != 'PtpClock':
        warnings.warn('Clock source of periodic signal is not `PtpClock`')
    camera.BslPeriodicSignalPeriod.SetValue(1 / framerate * 1e6)
    camera.BslPeriodicSignalDelay.SetValue(0)
    camera.TriggerSelector.SetValue("FrameStart")
    camera.TriggerMode.SetValue("On")
    camera.TriggerSource.SetValue("PeriodicSignal1")

    # # Transport Layer Control
    # camera.GevSCPD.SetValue(222768)
    # camera.GevSCFTD.SetValue(8018*index)
    # camera.GevSCPSPacketSize.SetValue(8000)

# Define your image handler class
class ImageHandler(pylon.ImageEventHandler):
    def __init__(self, camera_index, video_writer):
        super().__init__()
        self.camera_index = camera_index
        self.video_writer = video_writer

    def OnImageGrabbed(self, camera, grab_result):
        if grab_result.GrabSucceeded():
            image = grab_result.GetArray()
            self.video_writer.write(image)
        grab_result.Release()

# Create ImageHandler instances for both cameras
handler1 = ImageHandler(camera_index=0, video_writer=out1)
handler2 = ImageHandler(camera_index=1, video_writer=out2)

# Get the transport layer factory
tlFactory = pylon.TlFactory.GetInstance()

# Get all attached devices and exit application if no device is found
devices = tlFactory.EnumerateDevices()
cam_count = len(devices)
if not cam_count:
    raise EnvironmentError('No camera device found')

# Create and attach all Pylon Devices
cameras = pylon.InstantCameraArray(cam_count)
for camera, device in zip(cameras, devices):
    print('Using {:s} @ {:s}'.format(device.GetModelName(), device.GetIpAddress()))
    camera.Attach(tlFactory.CreateDevice(device))
    camera.Open()

# Initialize PTP and check initialization
for i, camera in enumerate(cameras):
    init_ptp(camera, i)

success = check_ptp(cameras)
if not success:
    raise EnvironmentError('PTP initialization was not successful')

# Initialize general camera parameters
for i, camera in enumerate(cameras):
    init_camera(camera, i)

# Image decompression
# decompressor = pylon.ImageDecompressor()
# descriptor = cameras[0].BslImageCompressionBCBDescriptor.GetAll()
# decompressor.SetCompressionDescriptor(descriptor)

# Prepare image grabbing
imgs = [None] * cam_count
ids = [None] * cam_count

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# Create smaller OpenCV windows for both camera streams
cv2.namedWindow('Camera 1 Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera 1 Stream', 960, 600)  # Adjust the size as needed

cv2.namedWindow('Camera 2 Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera 2 Stream', 960, 600)  # Adjust the size as needed

# Register image event handlers
for camera, handler in zip(cameras, [handler1, handler2]):
    camera.RegisterImageEventHandler(handler, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_Delete)

# cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByInstantCamera)

start_time = time.time()

while cameras.IsGrabbing():

    if cv2.waitKey(1) & 0xFF == 27:
        break

cameras.StopGrabbing()
cameras.Close()
out1.release()
out2.release()

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
frame_rate1 = num_frames1/elapsed_time
frame_rate2 = num_frames2/elapsed_time

print(f"Recording finished. Elapsed time: {elapsed_time:.2f} seconds")
print(f"Number of frames acquired (Camera 1): {num_frames1}")
print(f"Number of frames acquired (Camera 2): {num_frames2}")
print(f"Frame rate (camera1) is: {frame_rate1}")
print(f"Frame rate (camera2) is: {frame_rate2}")