from conf import *
from pypylon import pylon
from datetime import datetime

import cv2
import os
import numpy as np


class CameraCapture:
    
    def print_camera_info(self):
        # Get the camera info and print it
        info = self.camera.GetDeviceInfo()
        print(''.join(['-']*50))
        print(f"Camera information: {info.GetModelName()}")
        print(f"Camera serial number: {info.GetSerialNumber()}")
        print(f"Camera device version: {info.GetDeviceVersion()}")
        print(f"Camera device class: {info.GetDeviceClass()}")
        print(''.join(['-']*50))
    
    
    def __init__(self):
        self.camera = None
        self.open_camera()
        self.converter = pylon.ImageFormatConverter()
        # Setting the converter to output mono8 images for simplicity
        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.output_dim = [self.camera.Width.GetValue(), self.camera.Height.GetValue()]
        
        # Ensure the camera exposure, gain, and gamma are set to manual mode before adjusting
        self.camera.ExposureAuto.SetValue('Off')  # Turn off auto exposure
        self.camera.GainAuto.SetValue('Off')      # Turn off auto gain
        self.camera.GammaEnable.SetValue(True)    # Enable gamma correction if supported
        
        # Adjust camera settings - these values are examples and should be adjusted based on your needs and camera capabilities
        self.camera.ExposureTimeRaw.SetValue(80000)  # Set exposure time to 20000 microseconds
        self.camera.GainRaw.SetValue(0)            # Set gain
        self.camera.Gamma.SetValue(1.0)              # Set gamma value to 1.0 (if supported)
        self.print_camera_info()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
    def open_camera(self):
        if self.camera is not None:
            try:
                self.camera.Close()
            except Exception:
                pass
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
    
    def capture(self):
        while True:  # Change this to a more reliable condition if necessary
            if not self.camera.IsGrabbing():
                self.open_camera()
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            try:
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    # Convert to OpenCV format
                    image = self.converter.Convert(grabResult)
                    img = image.GetArray()
                    yield img
                grabResult.Release()
            except Exception as e:
                print("Error encountered: ", e)
                # Optionally, attempt to reconnect or handle error
                img = cv2.putText(np.zeros((self.output_dim[1], self.output_dim[0]), np.uint8),
                                  "No Image Input", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                yield img
                # No need to break here; let it attempt to reconnect in the next iteration

    def close(self):
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        self.camera.Close()



def image_resize(img, scale_percent):
    # Calculate the new dimensions based on a scaling factor (e.g., 50% of the original size)
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv2.resize(img, dim) 



def calculate_image_sharpness(image):
    # Compute gradients along the x and y axis, respectively
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # Compute the square root of the sum of the squared gradients
    focus_measure = np.sqrt(gx**2 + gy**2).mean()
    return focus_measure


def calculate_maximum_intensity(image):
    # Return the maximum intensity of the image
    return np.max(image)


# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global click_position 
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = (x, y)



# Display the image with the red box and magnified area
def display_image(save_to='', scale_factor=0.65):
    
    coupling = processing.IntensityMonitor()
    
    parent_directory = os.path.dirname(os.path.realpath(__file__))
    camera_capture = CameraCapture()
    cv2.namedWindow('Camera Output')
    cv2.setMouseCallback('Camera Output', mouse_callback)
    
    cv2.createTrackbar('Exposure time (ms)', 'Camera Output', 50, 500, 
                       lambda x: camera_capture.camera.ExposureTimeRaw.SetValue(x*1000))  # miniseconds
    
    try:
        for img in camera_capture.capture():
            resized_img = image_resize(img, scale_factor)  # only for display, not for saving
            sharpness = calculate_maximum_intensity(resized_img)
            if click_position:
                # Parameters for the red box and magnified area
                box_size = 170  # Size of the red box
                magnification = 4  # Magnification factor for the cropped area
                #click_position = (int(click_position[0] * (1+scale_factor)), int(click_position[1] * (1+scale_factor)))
                
                # Calculate the top left and bottom right points for the red box
                x1 = max(click_position[0] - box_size // 2, 0)
                y1 = max(click_position[1] - box_size // 2, 0)
                x2 = min(click_position[0] + box_size // 2, resized_img.shape[1] - 1)
                y2 = min(click_position[1] + box_size // 2, resized_img.shape[0] - 1)
                
                # Draw the red box
                cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Crop and magnify the area within the red box
                cropped_img = resized_img[y1:y2, x1:x2]
                magnified_img = cv2.resize(cropped_img, (0, 0), fx=magnification, fy=magnification)
                
                # Overlay the magnified image on the top left corner of the original image
                resized_img[0:magnified_img.shape[0], 0:magnified_img.shape[1]] = magnified_img
                
                sharpness = calculate_maximum_intensity(magnified_img)
            
            sharpness_text = f"Intensity: {sharpness:.2f}"
            
            intensity = coupling.add_image(resized_img)
            # Display the text on the image
            cv2.putText(resized_img, sharpness_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            
            
            cv2.imshow('Camera Output', utils.join_images([resized_img,intensity]))
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC key to exit
                break
            elif key == ord('s'):  # 's' key to save the image
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{parent_directory}/{save_to}/{timestamp}.png"
                cv2.imwrite(filename, img)
                print(f"Image saved as {filename}")
    finally:
        camera_capture.close()
        cv2.destroyAllWindows()


from ALP4 import *
DMD_DIM = 1024

if __name__ == "__main__":
    DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
    # calibration_img = simulation.generate_radial_gradient()
    calibration_img = np.ones((256, 256)) * 255
    # calibration_img = simulation.dmd_calibration_pattern_generation()
    calibration_img = simulation.macro_pixel(calibration_img, size=int(DMD_DIM/calibration_img.shape[0])) 
    DMD.display_image(dmd.dmd_img_adjustment(calibration_img, DMD_DIM)) # preload one image for camera calibration

    click_position = None
    display_image('results')


