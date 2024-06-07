from PIL import Image, ImageDraw, ImageFont

def create_counting_gif(filename, frame_duration, start=0, end=999):
    frames = []
    
    # Set a larger image size to enhance clarity with the default font
    image_size = (300, 150)
    
    # Use the default font provided by PIL
    font = ImageFont.load_default()

    for i in range(start, end + 1):
        image = Image.new('RGB', image_size, color=(255, 255, 255))
        d = ImageDraw.Draw(image)
        text = str(i)
        # Calculate text width and height using textbbox
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)
        textwidth = right - left
        textheight = bottom - top
        x = (image.width - textwidth) / 2
        y = (image.height - textheight) / 2
        d.text((x, y), text, fill=(0, 0, 0), font=font)
        frames.append(image)
    
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=frame_duration,
        loop=0
    )

create_counting_gif('../../ResultsCenter/counting_test_10ms.gif', frame_duration=10)




# class Camera(ABC):
#     def __init__(self, camera=None):
#         self.role = 'unsigned'  # reserved for PTP synchronization
#         self.camera = camera  # wrapped camera object of the specific camera brand
#         self.parameters = {}

#     @property
#     def role(self):
#         """
#         Get the role of the camera.
#         """
#         return self._role

#     @role.setter
#     def role(self, value):
#         """
#         Set the role of the camera with validation.
#         """
#         if value not in ['beam_image', 'fiber_output', 'unsigned']:
#             raise ValueError("Role must be 'beam_image' or 'fiber_output' or 'unsigned'.")
#         self._role = value
        
#     @abstractmethod
#     def set_parameter(self, **kwargs):
#         """
#         Set a camera parameter.
#         """
#         pass

#     @abstractmethod
#     def get_information(self):
#         """
#         Return a dictionary containing camera information.
#         """
#         pass
    
#     @abstractmethod
#     def demo(self):
#         """
#         freerun image streaming
#         """
#         pass


# class BaslerCamera(Camera):
#     def __init__(self, camera=None):
#         super().__init__(camera)
#         self.grabResult = None
        
#     def demo(self):
#         """
#         Generator method to continuously yield images as they are captured by the camera.
#         """
#         try:
#             self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser)
#             while self.camera.IsGrabbing():
#                 grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#                 if grabResult.GrabSucceeded():
#                     img = grabResult.GetArray()
#                     yield img
#                 grabResult.Release()
#         finally:
#             # Ensure the camera stops grabbing and properly releases resources when the generator stops.
#             self.camera.StopGrabbing()

#     def set_parameter(self, **kwargs):
#         pass

#     def get_information(self):
#         return {}
    
#     def refresh(self):
#         """
#         Refresh the camera (not restart Main purpose is to make sure the camera is ready for the next action command)
#         """
#         pass



    # def schedule_action_command(self, scheduled_time: int) -> np.ndarray:
    #     self.cameras[0].GevTimestampControlLatch.Execute() # Get the current timestamp from the camera
    #     current_time = self.cameras[0].GevTimestampValue.Value
    #     scheduled_time = current_time + scheduled_time  # Define the delay for action command (in nanoseconds)
    #     self._start_grabbing()
    #     # Issue the scheduled action command
    #     results = self.GigE_TL.IssueScheduledActionCommandNoWait(self.action_key, self.group_key, self.group_mask,
    #                                                              scheduled_time, self.boardcast_ip)
    #     print(f"Scheduled command issued at {int(scheduled_time//1e6)}ms later, retriving image...")
        
    #     grabResult0 = self.cameras[0].RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
    #     grabResult1 = self.cameras[1].RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
    #     if grabResult0.GrabSucceeded() & grabResult1.GrabSucceeded():
    #         im0 = grabResult0.GetArray()
    #         im1 = grabResult1.GetArray()
    #         t0 = grabResult0.TimeStamp
    #         t1 = grabResult1.TimeStamp
    #         timedif = self.max_time_difference([t0, t1])
    #         if timedif < 1000:
    #             combined_image = self._combine_images(im0, im1)
    #             print("Image retrived.")
    #         else: combined_image = None
            
    #     grabResult0.Release()
    #     grabResult1.Release()
    #     self._stop_grabbing() 
    #     return combined_image









    # def synchronization(self, threshold : int=300, timeout: int=20) -> list:
    #     self._check_cameras_ptp_state()
    #     print('Waiting for PTP time synchronization...')
    #     offset = float('inf')
    #     records = []
    #     start_time = time.time()
    #     while time.time() - start_time < timeout:
    #         offset = max(self.check_sync_status())
    #         records.append(offset)
    #         print(offset)
    #         offset = abs(offset)
    #         if offset < threshold: 
    #             print("Cameras synchronized.")
    #             return records
    #         time.sleep(1)
    #     raise TimeoutError("Cameras did not synchronize within the timeout period.")
    
    
    
    
    
    
    
    
    
    
    
    # def _check_cameras_ptp_state(self, timeout: int=10) -> bool:
    #     start_time = time.time()
    #     while time.time() - start_time < timeout:
    #         if all(camera.GevIEEE1588Status.Value in ['Slave', 'Master'] for camera in self.cameras):
    #             for i, camera in enumerate(self.cameras):
    #                 if camera.GevIEEE1588Status.Value == 'Master':
    #                     self.master = i
    #             return True
    #         time.sleep(0.5)
    #     raise TimeoutError("Cameras did not become ready within the timeout period.")
    
    # def initialize(self, timeout:int=10) -> None:
    #     """
    #     detect all cameras and initialize them
    #     """
    #     start_time = time.time()
    #     timeout += start_time
    #     while True:
    #         devices = self.tlFactory.EnumerateDevices()
    #         if len(devices) >= 2:
    #             break
    #         if time.time() >= timeout:
    #             raise RuntimeError(f"At least 2 cameras are required. Detected: {len(devices)}.")
    #         time.sleep(0.5)
        
    #     for i in range(len(devices)):  
    #         camera = pylon.InstantCamera(self.tlFactory.CreateDevice(devices[i]))
    #         camera.Open()
    #         camera.AcquisitionFrameRateEnable.Value = True
    #         camera.AcquisitionFrameRateAbs.Value = 20.0
    #         self.cameras.append(camera)
        
    #     self._flip_order()
        
    #     for i in self.cameras:  # prepare for PTP and scheduled action command
    #         i.AcquisitionFrameRateEnable.Value = False
    #         i.GevIEEE1588.Value = True
    #         i.AcquisitionMode.SetValue("SingleFrame") # SingleFrame Continuous
    #         i.TriggerMode.SetValue("On")
    #         i.TriggerSource.SetValue("Action1")
    #         i.TriggerSelector.SetValue('FrameStart')
    #         i.ActionDeviceKey.SetValue(self.action_key)
    #         i.ActionGroupKey.SetValue(self.group_key)
    #         i.ActionGroupMask.SetValue(self.group_mask)
    #     self.print_all_camera_status()