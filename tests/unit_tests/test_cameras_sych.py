from pypylon import pylon
import numpy as np
import cv2
from time import sleep
import matplotlib.pyplot as plt

def ptp_sync(cameras):
    # wait for PTP time synchronization and plot the offset results during synchronization
    offset = float('inf')
    temp = []
    while offset > 200:
        cameras[0].GevIEEE1588DataSetLatch.Execute()
        cameras[1].GevIEEE1588DataSetLatch.Execute()
        offset = cameras[i].GevIEEE1588OffsetFromMaster.Value
        print(offset)
        temp.append(offset)
        offset = abs(offset)
        sleep(1)
    plt.plot(temp, marker='o')  # 'o' creates a circle marker for each point
    # Adding title and labels
    plt.title('PTP Time sychronization between Cameras')
    plt.xlabel('Time')
    plt.ylabel('Offset from Master')
    # Displaying the plot
    plt.savefig('../../ResultsCenter/sync/timeshift.png')

def print_cam_status(cameras):
    for cam in cameras:
        info = cam.GetDeviceInfo()
        print('-' * 50)
        print("using %s @ %s @ %s" % (info.GetModelName(), info.GetSerialNumber(), info.GetIpAddress()))
        print("ActionGroupKey", hex(cam.ActionGroupKey.Value))
        print("ActionGroupMask", hex(cam.ActionGroupMask.Value))
        print("TriggerSource", cam.TriggerSource.Value)
        print("TriggerMode", cam.TriggerMode.Value)
        print("AcquisitionMode", cam.AcquisitionMode.Value)
        print('Camera grabing status: ', cam.IsGrabbing())
        print('-' * 50)
        print('\n')
        

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


# Shceduled action command testing
action_key = 0x4711
group_key = 0x1
group_mask = 0xffffffff # pylon.AllGroupMask


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
        
    # settings for the shceduled action command
    camera.TriggerSelector.SetValue('FrameStart')
    camera.TriggerMode.SetValue('On')
    camera.TriggerSource.SetValue('Action1')
    camera.AcquisitionMode.SetValue('Continuous') # SingleFrame, Continuous
    camera.ActionDeviceKey.SetValue(action_key)
    camera.ActionGroupKey.SetValue(group_key)
    camera.ActionGroupMask.SetValue(group_mask)


ptp_sync(cameras)


# Starts grabbing for all cameras
cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, 
                      pylon.GrabLoop_ProvidedByUser)
# cameras.StartGrabbing()

gige_tl = tlFactory.CreateTl('BaslerGigE')
print_cam_status(cameras)






# for counter in range(1, 5):
#     cameras[0].GevTimestampControlLatch.Execute()
#     currentTimestamp = cameras[0].GevTimestampValue.Value
#     actionTime = currentTimestamp - 1000 # action after 1 seconds from current timestamp
#     # use waiting variant on even counter
#     print(f'issuing action command with waiting for response, action at timestamp: {actionTime}, current timestamp: {currentTimestamp}')
#     timeout_ms = 1000
#     expected_results = 1
#     ok, results = gige_tl.IssueScheduledActionCommandWait(deviceKey=action_key, groupKey=group_key, groupMask=group_mask, actionTimeNs=actionTime, 
#                                     broadcastAddress="255.255.255.255", timeoutMs=5000, pNumResults=1)
#     print('action command results')
    
#     assert ok
#     for addr, status in results:
#         print(addr, act_cmd_status_strings[status])
    
#     grabResult1 = cameras[0].RetrieveResult(10000, 
#                           pylon.TimeoutHandling_ThrowException)
#     grabResult2 = cameras[1].RetrieveResult(10000, 
#                          pylon.TimeoutHandling_ThrowException)
    
#     if grabResult1.GrabSucceeded() & grabResult2.GrabSucceeded():
#         im1 = grabResult1.GetArray()
#         im2 = grabResult2.GetArray()
#         timediif = abs((grabResult1.TimeStamp - grabResult2.TimeStamp) * 1e-6)
#         print(timediif)
#         if timediif < 50:
#             filename = f"{'../../ResultsCenter/sync'}/{counter}_{timediif}.png"
#             combined_image = np.hstack((im1, im2))
#             cv2.imwrite(filename, combined_image)
#             print("timestamp difference (ms): ", timediif)
        








# cv2.namedWindow('Acquisition', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Acquisition', 1280, 512)

# while cameras.IsGrabbing():
#     grabResult1 = cameras[0].RetrieveResult(5000, 
#                          pylon.TimeoutHandling_ThrowException)
    
#     grabResult2 = cameras[1].RetrieveResult(5000, 
#                          pylon.TimeoutHandling_ThrowException)
    
#     if grabResult1.GrabSucceeded() & grabResult2.GrabSucceeded():
#         im1 = grabResult1.GetArray()
#         im2 = grabResult2.GetArray()
#         timestamp1 = grabResult1.TimeStamp
#         timestamp2 = grabResult2.TimeStamp
#         timediif = abs((timestamp2 - timestamp1) * 1e-6)
    
#         # If ESC is pressed exit and destroy window
#         combined_image = np.hstack((im1, im2))
#         cv2.imshow('Acquisition', combined_image)
#         key = cv2.waitKey(1)
#         if key == 27:
#             break
#         elif key == ord('s'):  # 's' key
#             if timediif < 10:
#                 filename = f"{'../../ResultsCenter/sync'}/{timestamp1}_{timestamp2}.png"
#                 cv2.imwrite(filename, combined_image)
#                 print("timestamp difference (ms): ", timediif)
#             else:
#                 print("not saved! timestamp difference (ms): ", timediif)
        
# cv2.destroyAllWindows()

cameras.StopGrabbing()
cameras.Close()

print("camera closed, program terminated!")
