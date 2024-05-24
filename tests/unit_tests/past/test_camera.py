from pypylon import pylon
import numpy as np
import cv2


cv2.namedWindow('Acquisition', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Acquisition', 1280, 512)


tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()
if len(devices) == 0:
    raise pylon.RUNTIME_EXCEPTION("No camera present.")


camera = pylon.InstantCamera(tlFactory.CreateFirstDevice())
camera.Open()
camera.AcquisitionFrameRateEnable.Value = True
camera.AcquisitionFrameRateAbs.Value = 20.0



# Starts grabbing for all cameras
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, 
                      pylon.GrabLoop_ProvidedByUser)


while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, 
                         pylon.TimeoutHandling_ThrowException)
    
    if grabResult.GrabSucceeded():
        img = grabResult.GetArray()
        cv2.imshow('Acquisition', img)
        key = cv2.waitKey(1)
        if key == 27:
            break

cv2.destroyAllWindows()




