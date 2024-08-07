# This sample demonstrates how to use action commands on a GigE camera to
# trigger images. Since this feature requires configuration of several camera
# features, this configuration is encapsuled in a dedicated configuration
# event handler called ActionTriggerConfiguration, whose usage is also
# demonstrated.

from pypylon import pylon
import numpy as np
import cv2
import time

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



tl_factory = pylon.TlFactory.GetInstance()

cam = None
for dev_info in tl_factory.EnumerateDevices():
    if dev_info.GetDeviceClass() == 'BaslerGigE':
        print("using %s @ %s" % (dev_info.GetModelName(), dev_info.GetIpAddress()))
        cam = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
        break
else:
    raise EnvironmentError("no GigE device found")


# Values needed for action commands. See documentation for the meaning of these
# values: https://docs.baslerweb.com/index.htm#t=en%2Faction_commands.htm
# For this simple sample we just make up some values.

action_key = 0x4711
group_key = 0x112233
group_mask = pylon.AllGroupMask

# Initiate automatic configuration by registering ActionTriggerConfiguration.
cam.RegisterConfiguration(
    pylon.ActionTriggerConfiguration(action_key, group_key, group_mask),
    pylon.RegistrationMode_Append,
    pylon.Cleanup_Delete
    )
cam.Open()

# Demonstrate effect of ActionTriggerConfiguration by printing out those values
# affected by it.
# print("ActionDeviceKey", hex(cam.ActionDeviceKey.Value))
print("ActionGroupKey:", hex(cam.ActionGroupKey.Value))
print("ActionGroupMask:", hex(cam.ActionGroupMask.Value))
print("TriggerSource:", cam.TriggerSource.Value)
print("TriggerMode:", cam.TriggerMode.Value)
print("AcquisitionMode:", cam.AcquisitionMode.Value)
print('\n')

# Create a suitable ActionCommand object. For that a GigETransportLayer object
# is needed.
gige_tl = tl_factory.CreateTl('BaslerGigE')

# Using default value of "255.255.255.255" for fourth
# parameter 'broadcastAddress'.

# act_cmd = gige_tl.ActionCommand(action_key, group_key, group_mask)
cam.GevTimestampControlLatch.Execute()
currentTimestamp = cam.GevTimestampValue.Value
actionTime = 0 # currentTimestamp + 500000000
act_cmd = gige_tl.ScheduledActionCommand(action_key, group_key, group_mask, actionTimeNs=actionTime)

print(type(act_cmd))


cam.StartGrabbing()

for counter in range(1, 5):

    # Issue action command
    if counter & 1:
        # use no-wait variant on odd counter
        print('issuing no-wait action command')
        ok = act_cmd.IssueNoWait()
        assert ok
    else:
        # use waiting variant on even counter
        print('issuing action command with waiting for response')
        timeout_ms = 10000
        expected_results = 1
        ok, results = act_cmd.IssueWait(timeout_ms, expected_results)
        print('action command results')
        assert ok
        for addr, status in results:
            print(addr, act_cmd_status_strings[status])


    # while 1:
    #     try:
    #         cam.RetrieveResult(10, pylon.TimeoutHandling_ThrowException)
    #         if grabResult.GrabSucceeded(): print('grab succeeded!!!!!!!!!!!')
    #     except pylon.TimeoutException:
    #         pass
    #     cam.GevTimestampControlLatch.Execute()
    #     timestamp = cam.GevTimestampValue.Value
    #     print(f'{actionTime - timestamp} action time: {actionTime} grab result at timestamp {timestamp}: failed')
    #     if actionTime - timestamp < 0: break
            
    print(f'retrieving frame at timestamp: {actionTime}, current timestamp: {cam.GevTimestampValue.Value}')
    with cam.RetrieveResult(100000, pylon.TimeoutHandling_ThrowException) as grabResult:
        print("received frame %d\n" % counter)
        img = grabResult.GetArray()
        filename = f"{'../../ResultsCenter/sync'}/{counter}.png"
        cv2.imwrite(filename, img)
    
cam.StopGrabbing()
cam.Close()