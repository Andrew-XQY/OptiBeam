from conftest import *
from datetime import datetime
import cv2
from pypylon import pylon
        
     

CAM_INFO = {"ground_truth": "24548598", "speckle_pattern": "24876080"}

tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()

if len(devices) == 0:
    print("No cameras detected.")
else:
    print("Number of cameras detected:", len(devices))
    # Create a list to hold the camera objects
    cameras = []
    # Create an InstantCamera object for each detected device and add it to the cameras list
    for device in devices:
        cameras.append(pylon.InstantCamera(tl_factory.CreateDevice(device)))
        
for i in cameras:
    print(f"Camera attached: {i.GetDeviceInfo().GetModelName()} - Serial Number: {i.GetDeviceInfo().GetSerialNumber()}")
    
camera1 = camera.BaslerCamera(cameras[0])
camera2 = camera.BaslerCamera(cameras[1])


camera1.camera.GevIEEE1588.Value = True

# print("done")


from threading import Thread
import queue

def fetch_images(camera, output_queue):
    for img in camera.capture():
        output_queue.put(img)

# Assuming camera1 and camera2 are your camera instances
output_queue1 = queue.Queue()
output_queue2 = queue.Queue()

# Starting separate threads to handle image fetching
thread1 = Thread(target=fetch_images, args=(camera1, output_queue1))
thread2 = Thread(target=fetch_images, args=(camera2, output_queue2))
thread1.start()
thread2.start()

save_to = "C:\\Users\\qiyuanxu\\Documents\\ResultsCenter\\images"
try:
    while True:
        if not output_queue1.empty() and not output_queue2.empty():
            img1 = output_queue1.get()
            img2 = output_queue2.get()

            # Assuming both images are of the same size and gray scale for simplicity
            combined_image = np.hstack((img1, img2))

            # Display the stitched image
            combined_image = cv2.resize(combined_image, (1200, 600))
            cv2.imshow('Combined Stream', combined_image)

            key = cv2.waitKey(1)
            if key == 27:  # ESC key to exit
                break
            elif key == ord('s'):  # 's' key to save the image
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{save_to}/{timestamp}.png"
                cv2.imwrite(filename, combined_image)
                print(f"Image saved as {filename}")
finally:
    cv2.destroyAllWindows()
    # Stop the capture
    camera1.camera.StopGrabbing()
    camera2.camera.StopGrabbing()
    # Wait for threads to finish
    thread1.join()
    thread2.join()
    
    
    
    
    
    
    
    
    
    
    
# Create camera objects for the first two detected devices
# camera1 = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
# camera2 = pylon.InstantCamera(tl_factory.CreateDevice(devices[1]))

# CMOS = camera.BaslerCamera(camera2)
# CMOS.demo_video()






















