import numpy as np
import cv2
from pypylon import pylon
from conftest import *

# get the Basler cameras resource manager
cam_manager = pylon.TlFactory.GetInstance()

# using the resource manager, get the list of available (detected) cameras
devices = cam_manager.EnumerateDevices()

if len(devices) == 0:
    raise "No cameras detected."
else:
    print("Number of cameras detected:", len(devices))
    cameras = []
    for d in devices:
        cam = camera.BaslerCamera(pylon.InstantCamera(cam_manager.CreateDevice(d)))
        cameras.append(cam)
        
# Set the PTP on for all cameras internal clock synchronization
for c in cameras:
    c.camera.GevIEEE1588.Value = True
    c.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
print("PTP enabled for all cameras")


cam = cameras[1].camera
cam.GevIEEE1588DataSetLatch 
temp = cam.PtpServoStatus.Value 
print(temp)







# # Start grabbing images from all cameras
# while True:
#     imgs = []
#     timestamps = []
#     for cam in cameras:
#         if cam.camera.IsGrabbing():
#             grab_result = cam.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#             if grab_result.GrabSucceeded():
#                 img = grab_result.Array
#                 timestamp = grab_result.TimeStamp
#                 imgs.append(img)
#                 timestamps.append(timestamp)
#             grab_result.Release()
    
#     # Display the images and save
#     combined_image = np.hstack((imgs[0], imgs[1]))
#     combined_image = cv2.resize(combined_image, (960, 300))
#     cv2.imshow('Camera View', combined_image)  # Display the first camera image
#     key = cv2.waitKey(1)
#     if key == 27:  # ESC key
#         break
#     elif key == ord('s'):  # 's' key
#         for i, (img, timestamp) in enumerate(zip(imgs, timestamps), 1):
#             filename = f"{'../../ResultsCenter/sync'}/{timestamp}_cam_{i}.png"
#             cv2.imwrite(filename, img)
#             print(f"Saved {filename}")


# # Cleanup
# cv2.destroyAllWindows()



for cam in cameras:
    cam.close()




