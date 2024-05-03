from pypylon import genicam
from pypylon import pylon
import sys
import os
import cv2

# initalize manager and cameras container
tl_factory  = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
cameras = pylon.InstantCameraArray(len(devices))

save_path = '../../ResultsCenter/sync'
time_to_start = 10000000000 # Set the desired action time in nanoseconds
action_key = 0x4711
group_key = 0x112233
group_mask = pylon.AllGroupMask


 # Create and attach all Pylon Devices. (register cameras to the container)
for i, cam in enumerate(cameras):
    cam.Attach(tl_factory.CreateDevice(devices[i]))
    # Initiate automatic configuration by registering ActionTriggerConfiguration.
    cam.RegisterConfiguration(
        pylon.ActionTriggerConfiguration(action_key, group_key, group_mask),
        pylon.RegistrationMode_Append,
        pylon.Cleanup_Delete
        )
    cam.Open()
    cam.ExposureTimeRaw.SetValue(5000)

    # Demonstrate effect of ActionTriggerConfiguration by printing out those values
    # affected by it.
    print("-"*50)
    # print("ActionDeviceKey", hex(cam.ActionDeviceKey.Value))
    print("ActionGroupKey", hex(cam.ActionGroupKey.Value))
    print("ActionGroupMask", hex(cam.ActionGroupMask.Value))
    print("TriggerSource", cam.TriggerSource.Value)
    print("TriggerMode", cam.TriggerMode.Value)
    print("AcquisitionMode", cam.AcquisitionMode.Value)
    print("-"*50)



# Create a suitable ActionCommand object. For that a GigETransportLayer object
# is needed.
gige_tl = tl_factory.CreateTl('BaslerGigE')

# Using default value of "255.255.255.255" for fourth
# parameter 'broadcastAddress'.
act_cmd = gige_tl.ActionCommand(action_key, group_key, group_mask)

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


for i, c in enumerate(cameras):
    cam = c
    
cameras.StartGrabbing()

for counter in range(1, 9):

    # Issue action command
    if counter & 1:
        # use no-wait variant on odd counter
        print('issuing no-wait action command')
        ok = act_cmd.IssueNoWait()
        assert ok
    else:
        # use waiting variant on even counter
        print('issuing action command with waiting for response')
        timeout_ms = 1000
        expected_results = 1
        ok, results = act_cmd.IssueWait(timeout_ms, expected_results)
        print('action command results')
        assert ok
        for addr, status in results:
            print(addr, act_cmd_status_strings[status])

    with cameras.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException) as grab_result:
        if grab_result.GrabSucceeded():
            #img = grab_result.Array
            cv_image = cv2.cvtColor(grab_result.GetArray(), cv2.COLOR_BAYER_RG2RGB)
            cv2.imwrite(os.path.join(save_path, f"camera_{0}_image_{grab_result.BlockID}.png"), cv_image)
        print("received frame %d\n" % counter)

cameras.StopGrabbing()
cameras.Close()














# for (size_t i = 0; i > cameras.GetSize(); ++i)
# {
#     // Open the camera connection
#     cameras[i].Open();
#     // Configure the trigger selector
#     cameras[i].TriggerSelector.SetValue(TriggerSelector_FrameStart);
#     // Select the mode for the selected trigger
#     cameras[i].TriggerMode.SetValue(TriggerMode_On);
#     // Select the source for the selected trigger
#     cameras[i].TriggerSource.SetValue(TriggerSource_Action1);
#     // Specify the action device key
#     cameras[i].ActionDeviceKey.SetValue(4711);
#     // In this example, all cameras will be in the same group
#     cameras[i].ActionGroupKey.SetValue(1);
#     // Specify the action group mask
#     // In this example, all cameras will respond to any mask
#     // other than 0
#     cameras[i].ActionGroupMask.SetValue(0xffffffff);
# }

# //--- End of camera setup ---
# // Get the current timestamp of the first camera
# // NOTE: All cameras must be synchronized via Precision Time Protocol
# camera[0].GevTimestampControlLatch.Execute();
# int64_t currentTimestamp = camera[0].GevTimestampValue.GetValue();
# // Specify that the command will be executed roughly 30 seconds
# // (30 000 000 000 ticks) after the current timestamp.
# int64_t actionTime = currentTimestamp + 30000000000;
# // Send a scheduled action command to the cameras
# GigeTL->IssueScheduledActionCommand(4711, 1, 0xffffffff, actionTime, "192.168.1.255");



# save_path = '../../ResultsCenter/sync'

# # Open cameras again for acquisition
# for cam in cameras:
#     cam.Open()
#     cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# try:
#     while cameras.IsGrabbing():
#         grab_result = cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#         if grab_result.GrabSucceeded():
#             # Convert grabbed image to an OpenCV format
#             img = grab_result.Array
#             cv_image = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)  # Adjust color conversion as needed
#             # Save image
#             cv2.imwrite(os.path.join(save_path, f"camera_{i}_image_{grab_result.BlockID}.png"), cv_image)
#             print(f"Saved camera {i} image {grab_result.BlockID}.")
#         grab_result.Release()
# finally:
#     # Stopping and closing cameras
#     for cam in cameras:
#         cam.StopGrabbing()
#         cam.Close()

# print("Acquisition stopped.")

















