"""
after the data collection, carry out the post-processing of the data, update thoes metadata to the database
including 1. cropping images positions
2. beam parameters
"""

from conf import *
import cv2
import pandas as pd

# Database Initialization
DB = database.SQLiteDB(DATABASE_ROOT)

# Select the images to process (could based on other fields, adjust the query accordingly)
image_selected = DB.sql_select("SELECT id, image_path FROM mmf_dataset_metadata WHERE batch=1")

# ----------------- Update crop position -----------------
image_contained = 2
crop_areas = processing.select_crop_areas_center(cv2.imread(image_selected["image_path"][-1]), num=image_contained, scale_factor=(1/image_contained)) 
print("Defined crop areas:", crop_areas)

original_crop = [crop_areas[0]]*len(image_selected)
fiber_crop = [crop_areas[1]]*len(image_selected)
df = pd.DataFrame({"id": image_selected["id"], "original_crop_pos": original_crop, "speckle_crop_pos": fiber_crop})
sql = DB.batch_update("mmf_dataset_metadata", "id", df)
DB.sql_execute(sql)

# ----------------- Update beam parameters -----------------
params = []

for path in image_selected["image_path"]:
    img = utils.load_image(path)
    params.append(evaluation.beam_params(img))
    
df = pd.DataFrame({"id": image_selected["id"], "beam_parameters": params})
sql = DB.batch_update("mmf_dataset_metadata", "id", df)
DB.sql_execute(sql)


# Close the database connection
DB.close()
