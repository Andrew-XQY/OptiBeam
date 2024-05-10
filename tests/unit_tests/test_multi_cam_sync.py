from pypylon import pylon
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# ----------------- define functions -----------------
def ptp_sync(cameras, slave):
    # wait for PTP time synchronization and plot the offset results during synchronization
    offset = float('inf')
    temp = []
    while offset > 200:
        # cameras.GevIEEE1588DataSetLatch.Execute()
        cameras[0].GevIEEE1588DataSetLatch.Execute()
        cameras[1].GevIEEE1588DataSetLatch.Execute()
        offset = cameras[slave].GevIEEE1588OffsetFromMaster.Value
        print(offset)
        temp.append(offset)
        offset = abs(offset)
        time.sleep(1)
    plt.plot(temp, marker='o')  
    plt.title('PTP Time sychronization between Cameras')
    plt.xlabel('Time')
    plt.ylabel('Offset from Master')
    plt.savefig('../../ResultsCenter/sync/timeshift.png')
    
def print_cam_status(cameras):
    for cam in cameras:
        info = cam.GetDeviceInfo()
        print('-' * 50)
        print("Using %s @ %s @ %s" % (info.GetModelName(), info.GetSerialNumber(), info.GetIpAddress()))
        print("ActionGroupKey:", hex(cam.ActionGroupKey.Value))
        print("ActionGroupMask:", hex(cam.ActionGroupMask.Value))
        print("TriggerSource:", cam.TriggerSource.Value)
        print("TriggerMode:", cam.TriggerMode.Value)
        print("AcquisitionMode:", cam.AcquisitionMode.Value)
        print('Camera grabing status: ', cam.IsGrabbing())
        print('Camera PTP status: ', cam.GevIEEE1588Status.Value)
        print('-' * 50)
        print('\n')

# ----------------- define variables -----------------
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

action_key = 0x1
group_key = 0x1
group_mask = pylon.AllGroupMask # pylon.AllGroupMask or 0xffffffff



# ----------------- multiple cameras initialization -----------------
tlFactory = pylon.TlFactory.GetInstance()
gige_tl = tlFactory.CreateTl('BaslerGigE') # GigE transport layer, used for issuing action commands later
devices = tlFactory.EnumerateDevices()
cameras = pylon.InstantCameraArray(2)

for i, camera in enumerate(cameras):
    camera.Attach(tlFactory.CreateDevice(devices[i]))
    camera.Open()
    camera.GevIEEE1588.Value = True
    if camera.GevIEEE1588Status.Value == 'Slave': slave = i
    camera.AcquisitionMode.SetValue("SingleFrame") # SingleFrame Continuous
    camera.TriggerMode.SetValue("On")
    camera.TriggerSource.SetValue("Action1")
    camera.TriggerSelector.SetValue('FrameStart')
    camera.ActionDeviceKey.SetValue(action_key)
    camera.ActionGroupKey.SetValue(group_key)
    camera.ActionGroupMask.SetValue(group_mask)

ptp_sync(cameras, slave)
cameras.StartGrabbing()
print_cam_status(cameras)


# ----------------- issue scheduled action command -----------------
for _ in range(5):
    camera.GevTimestampControlLatch.Execute() # Get the current timestamp from the camera
    current_time = camera.GevTimestampValue.Value
    scheduled_time = current_time + 3000000000  # Define the delay for action command (in nanoseconds)

    # Issue the scheduled action command
    results = gige_tl.IssueScheduledActionCommandNoWait(action_key, group_key, 
                                                        group_mask, scheduled_time, "255.255.255.255")
    print(f"Scheduled command issued, scheduled capture at {scheduled_time}, retriving image...")
    
    # Wait for the grab result to ensure images are captured after trigger
    grabResult1 = cameras[0].RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
    grabResult2 = cameras[1].RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)

    if grabResult1.GrabSucceeded() & grabResult2.GrabSucceeded():
        im1 = grabResult1.GetArray()
        im2 = grabResult2.GetArray()
        t1 = grabResult1.TimeStamp
        t2 = grabResult2.TimeStamp
        timediif = abs((t1 - t2))
        print(f"Camera 1 image captured at scheduled time      {t1}, time difference: {t1 - scheduled_time} ns")
        print(f"Camera 2 image captured at scheduled time      {t2}, time difference: {t2 - scheduled_time} ns")
        print(f"Time difference between two images: {timediif} ns \n")
        if timediif < 1000:
            filename = f"{'../../ResultsCenter/sync'}/{scheduled_time}.png"
            combined_image = np.hstack((im1, im2))
            cv2.imwrite(filename, combined_image)

    # Release the grab
    grabResult1.Release()
    grabResult2.Release()
    cameras.StopGrabbing()
    cameras.StartGrabbing()


# ----------------- close cameras -----------------
cameras.StopGrabbing()
cameras.Close()
print("Camera closed, program terminated!")
