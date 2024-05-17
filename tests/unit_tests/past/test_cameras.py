from pypylon import pylon
import numpy as np
import cv2


cv2.namedWindow('Acquisition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Acquisition', 1280, 512)


tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()
if len(devices) == 0:
    raise pylon.RUNTIME_EXCEPTION("No camera present.")

cameras = pylon.InstantCameraArray(2)

for i, camera in enumerate(cameras):
    camera.Attach(tlFactory.CreateDevice(devices[i]))
    camera.Open()
    # Set the acquisition frame rate
    # camera.AcquisitionFrameRateEnable.Value = True
    # camera.AcquisitionFrameRateAbs.Value = 60.0 # Set frame rate in fps
    # camera.ExposureAuto.SetValue('Off')
    # camera.ExposureTimeRaw.SetValue(1000)  # Set exposure time in microseconds
    camera.GevIEEE1588.Value = True
    print(camera.GevIEEE1588Status.Value)
    if camera.GevIEEE1588Status.Value == 'Slave':
        slave = i


# Starts grabbing for all cameras
cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, 
                      pylon.GrabLoop_ProvidedByUser)


while cameras.IsGrabbing():
    grabResult1 = cameras[0].RetrieveResult(5000, 
                         pylon.TimeoutHandling_ThrowException)
    
    grabResult2 = cameras[1].RetrieveResult(5000, 
                         pylon.TimeoutHandling_ThrowException)
    
    if grabResult1.GrabSucceeded() & grabResult2.GrabSucceeded():
        im1 = grabResult1.GetArray()
        im2 = grabResult2.GetArray()
        timestamp1 = grabResult1.TimeStamp
        timestamp2 = grabResult2.TimeStamp

        timediif = abs((timestamp2 - timestamp1) * 1e-6)
        # If ESC is pressed exit and destroy window
        combined_image = np.hstack((im1, im2))
        cv2.imshow('Acquisition', combined_image)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('s'):  # 's' key
            if timediif < 10:
                filename = f"{'../../ResultsCenter/sync'}/{timestamp1}_{timestamp2}.png"
                cv2.imwrite(filename, combined_image)
                print("timestamp difference (in ms): ", timediif)
            else:
                print("Not saved, because timestamp difference (in ms): ", timediif)

cv2.destroyAllWindows()




