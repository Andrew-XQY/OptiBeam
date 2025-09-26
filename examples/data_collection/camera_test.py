"""
Camera unit test script - extracted from data_collection_dmd.py
This script focuses purely on camera functionality testing without DMD or database operations.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime, time
import cv2, json
from tqdm import tqdm
import traceback
from contextlib import ContextDecorator


class ChangeDirToFileLocation(ContextDecorator):
    def __enter__(self):
        # Save the current working directory
        self.original_cwd = os.getcwd()
        # Change the working directory to the location of the currently running .py file
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset the working directory back to its original location
        os.chdir(self.original_cwd)


with ChangeDirToFileLocation():
    full_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    sys.path.insert(0, full_path)
    import optibeam.camera as camera
    import optibeam.processing as processing
    import optibeam.analysis as analysis
    import optibeam.utils as utils


# ============================
# Camera Test Parameters
# ============================
conf = {
    'camera_order_flip': True,  # camera order flip
    'cam_schedule_time': int(300 * 1e6),  # camera schedule time in milliseconds
    'base_resolution': (256, 256),  # base resolution for all images
    'number_of_test_images': 10,  # number of test images to capture
    'crop_areas': [((868, 433), (1028, 593)), ((2762, 343), (3216, 797))],  # crop areas for the camera images
    'config_crop_area': False,  # set to True to configure crop areas interactively
}


# ============================
# Camera Initialization
# ============================
print("Initializing cameras...")
MANAGER = camera.MultiBaslerCameraManager()
if conf['camera_order_flip']: 
    MANAGER.flip = True
MANAGER.initialize()
MANAGER.synchronization()
print("Camera initialization complete.")


# ============================
# Select crop areas (optional step)
# ============================
if conf['config_crop_area']:
    print("Taking test image for crop area configuration...")
    test_img = MANAGER.schedule_action_command(conf['cam_schedule_time'])
    test_img = processing.add_grid(test_img, partitions=50)
    crop_areas = processing.select_crop_areas_corner(test_img, num=2, scale_factor=0.6) 
    print(f"Crop areas selected: {crop_areas}")
    print("Procedure completed. Please update the 'crop_areas' in the configuration.")
    MANAGER.end()
    sys.exit()


# ============================
# Camera Testing Pipeline
# ============================
# Create output directory for test images
output_dir = f"camera_test_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

try:
    print(f"Starting camera test - capturing {conf['number_of_test_images']} images...")
    
    for i in tqdm(range(conf['number_of_test_images']), desc="Capturing images"):
        # Capture image from cameras
        image = MANAGER.schedule_action_command(conf['cam_schedule_time'])
        
        if image is not None:
            # Generate filename with timestamp
            filename = f"camera_test_{i:04d}_{time.time_ns()}.png"
            image_path = os.path.join(output_dir, filename)
            
            # Save original full image
            cv2.imwrite(image_path, image)
            
            # Process and analyze the image
            if conf['crop_areas']:
                # Crop the image to regions of interest
                cropped_image = processing.crop_image_from_coordinates(image, conf['crop_areas'])
                
                # Split into ground truth and speckle parts
                ground_truth, speckle = utils.split_image(cropped_image)
                
                # Analyze both parts
                stats = {
                    'ground_truth_img': analysis.analyze_image(ground_truth),
                    'fiber_output_img': analysis.analyze_image(speckle)
                }
                
                # Save cropped image
                cropped_filename = f"camera_test_cropped_{i:04d}_{time.time_ns()}.png"
                cropped_path = os.path.join(output_dir, cropped_filename)
                cv2.imwrite(cropped_path, cropped_image)
                
                print(f"Image {i+1}: Original shape: {image.shape}, Cropped shape: {cropped_image.shape}")
                print(f"  Ground truth stats: {stats['ground_truth_img']}")
                print(f"  Fiber output stats: {stats['fiber_output_img']}")
            else:
                print(f"Image {i+1}: Shape: {image.shape}")
                stats = analysis.analyze_image(image)
                print(f"  Image stats: {stats}")
            
            # Small delay between captures
            time.sleep(0.1)
        else:
            print(f"Warning: Failed to capture image {i+1}")
    
    print(f"Camera test completed successfully. Images saved to: {output_dir}")

except Exception as e:
    print(f"An error occurred during camera testing: {e}")
    traceback.print_exc()

finally:
    # ============================
    # Clean up
    # ============================
    print("Closing camera connections...")
    MANAGER.end()
    print("Camera test finished.")