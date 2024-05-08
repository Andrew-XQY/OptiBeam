from pypylon import pylon
import numpy as np
import cv2
from time import sleep


# possible results for issuing an action command
act_cmd_status_strings = {
    pylon.GigEActionCommandStatus_Ok:
        'The device acknowledged the command',
    pylon.GigEActionCommandStatus_NoRefTime:
        'The device is not synchronized to a master clock',
    pylon.GigEActionCommandStatus_Overflow:
        'The action commands queue is full',
    pylon.GigEActionCommandStatus_ActionLate:
        'The requested action time was at a point in time that is in the past',
    }




# initialization of transport layer factory and cameras
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
        





    
# wait for PTP time synchronization and plot the offset results during synchronization
offset = float('inf')
temp = []
while offset > 100:
    cameras[0].GevIEEE1588DataSetLatch.Execute()
    cameras[1].GevIEEE1588DataSetLatch.Execute()
    offset = cameras[i].GevIEEE1588OffsetFromMaster.Value
    print(offset)
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

# Shceduled action command testing
cameras[0].GevTimestampControlLatch.Execute()
currentTimestamp = cameras[0].GevTimestampValue.Value
print("current master (camera0) timestamp: ", currentTimestamp)
actionTime = currentTimestamp + 5000000000 # action after 5 seconds from current timestamp

action_key = 0x4711
group_key = 0x112233
group_mask = pylon.AllGroupMask

gige_tl = tlFactory.CreateTl('BaslerGigE')
ok, results = gige_tl.IssueScheduledActionCommandWait(deviceKey=action_key, groupKey=group_key, groupMask=group_mask, actionTimeNs=actionTime, 
                                        broadcastAddress="192.168.1.255", timeoutMs=5000, pNumResults=1)

# ok, results = act_cmd.IssueWait(timeout_ms, expected_results)
print('action command results')
for addr, status in results:
    print(addr, act_cmd_status_strings[status])







cv2.namedWindow('Acquisition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Acquisition', 1280, 512)

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
                print("timestamp difference (ms): ", timediif)
            else:
                print("not saved! timestamp difference (ms): ", timediif)
        
    
cv2.destroyAllWindows()


