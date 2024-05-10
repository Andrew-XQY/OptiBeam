from pypylon import pylon
import time
import numpy as np
import cv2


action_key = 0x1
group_key = 0x1
group_mask = pylon.AllGroupMask # pylon.AllGroupMask   0xffffffff


# Initialize the camera
tlFactory = pylon.TlFactory.GetInstance()
gige_tl = tlFactory.CreateTl('BaslerGigE')

camera = pylon.InstantCamera(tlFactory.CreateFirstDevice())
camera.Open()

# Configure the camera for using an action command
camera.AcquisitionMode.SetValue("SingleFrame") # SingleFrame Continuous
camera.TriggerMode.SetValue("On")
camera.TriggerSource.SetValue("Action1")
camera.TriggerSelector.SetValue('FrameStart')
camera.ActionDeviceKey.SetValue(action_key)
camera.ActionGroupKey.SetValue(group_key)
camera.ActionGroupMask.SetValue(group_mask)
info = camera.GetDeviceInfo()
print("Using %s @ %s @ %s" % (info.GetModelName(), info.GetSerialNumber(), info.GetIpAddress()))
print("ActionGroupKey:", hex(camera.ActionGroupKey.Value))
print("ActionGroupMask:", hex(camera.ActionGroupMask.Value))
print("TriggerSource:", camera.TriggerSource.Value)
print("TriggerMode:", camera.TriggerMode.Value)
print("TriggerSelector:", camera.TriggerSelector.Value)
print("AcquisitionMode:", camera.AcquisitionMode.Value)

# Start grabbing
camera.StartGrabbing()
# camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
print('Camera grabing status: ', camera.IsGrabbing())
print('\n')
        

# Get the current timestamp from the camera (assuming nanosecond precision)
camera.GevTimestampControlLatch.Execute()
current_time = camera.GevTimestampValue.Value
# Define the delay for action command (5 seconds in nanoseconds)
scheduled_time = current_time + 3000000000



# Issue the scheduled action command
results = gige_tl.IssueScheduledActionCommandNoWait(action_key, group_key, group_mask, scheduled_time, "255.255.255.255")
print(f"command issued, scheduled capture at {scheduled_time}, retriving...")


# Wait for the grab result to ensure images are captured after trigger
grabResult = camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)

if grabResult.GrabSucceeded():
    # Process the grab result, for example, convert to an image
    img = grabResult.GetArray()
    timestamp = grabResult.TimeStamp
    print(f"Image captured at scheduled time     {timestamp}")
    filename = f"{'../../ResultsCenter/sync'}/{timestamp}.png"
    cv2.imwrite(filename, img)

# Release the result and close camera
grabResult.Release()
camera.StopGrabbing()
camera.Close()
