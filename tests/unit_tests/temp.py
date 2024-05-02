
import os
os.environ["PYLON_CAMEMU"] = "2"
import pypylon.pylon as pylon
import pypylon.genicam as geni
#import from samples folder
from configurationeventprinter import ConfigurationEventPrinter
from imageeventprinter import ImageEventPrinter
import time

class CameraBasler(object):

    def __init__(self):
        tlFactory = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RUNTIME_EXCEPTION("No camera present.")

        # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        self.cameras = pylon.InstantCameraArray(min(len(devices), 2))

        # Create and attach all Pylon Devices.
        for i, cam in enumerate(self.cameras):
            cam.Attach(tlFactory.CreateDevice(devices[i]))
            cam.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(), pylon.RegistrationMode_ReplaceAll,
                                      pylon.Cleanup_Delete)
            # For demonstration purposes only, add a sample configuration event handler to print out information
            # about camera use.t
            cam.RegisterConfiguration(ConfigurationEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
            # The image event printer serves as sample image processing.
            # When using the grab loop thread provided by the Instant Camera object, an image event handler processing the grab
            # results must be created and registered.
            cam.RegisterImageEventHandler(ImageEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
        # Start the grabbing using the grab loop thread, by setting the grabLoopType parameter
        # to GrabLoop_ProvidedByInstantCamera. The grab results are delivered to the image event handlers.
        # The GrabStrategy_OneByOne default grab strategy is used.
        self.cameras.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
        self.cameraL = self.cameras[0]
        self.cameraR = self.cameras[1]


    # The grabbing is stopped, the device is closed and destroyed automatically when the camera object goes out of scope.
    def takeStereo(self):
        if self.cameraL.WaitForFrameTriggerReady(100,
                                                 pylon.TimeoutHandling_ThrowException) and self.cameraR.WaitForFrameTriggerReady(
                100, pylon.TimeoutHandling_ThrowException):
            self.cameraL.ExecuteSoftwareTrigger()
            self.cameraR.ExecuteSoftwareTrigger()


cams=CameraBasler()
for i in range(10):
    cams.takeStereo()
    time.sleep(1)