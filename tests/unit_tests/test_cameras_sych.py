from pypylon import pylon
import numpy as np
import cv2
from time import sleep


cv2.namedWindow('Acquisition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Acquisition', 1280, 512)


tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()
cameras = pylon.InstantCameraArray(2)

for i, camera in enumerate(cameras):
    camera.Attach(tlFactory.CreateDevice(devices[i]))
    camera.Open()
    camera.GevIEEE1588.Value = True
    print(camera.GevIEEE1588Status.Value)
    if camera.GevIEEE1588Status.Value == 'Slave':
        slave = i
    
    
offset = float('inf')
temp = []
while offset > 30:
    cameras[0].GevIEEE1588DataSetLatch.Execute()
    cameras[1].GevIEEE1588DataSetLatch.Execute()
    
    test = cameras[0].GevTimestampValue.Value - cameras[1].GevTimestampValue.Value 
    offset = cameras[i].GevIEEE1588OffsetFromMaster.Value
    print(offset, test)
    temp.append(offset)
    offset = abs(offset)
    sleep(1)

import matplotlib.pyplot as plt
plt.plot(temp, marker='o')  # 'o' creates a circle marker for each point
# Adding title and labels
plt.title('PTP Time sychronization between Cameras')
plt.xlabel('Time')
plt.ylabel('Offset from Master')
# Displaying the plot
plt.savefig('../../ResultsCenter/sync/timeshift.png')



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
        timestamp1 = grabResult1.TimeStamp
        im2 = grabResult2.GetArray()
        timestamp2 = grabResult2.TimeStamp
    
        # If ESC is pressed exit and destroy window
        combined_image = np.hstack((im1, im2))
        cv2.imshow('Acquisition', combined_image)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('s'):  # 's' key
            filename = f"{'../../ResultsCenter/sync'}/{timestamp1}_{timestamp2}.png"
            cv2.imwrite(filename, combined_image)
            print("timestamp difference: ", timestamp2 - timestamp1)
    
cv2.destroyAllWindows()




