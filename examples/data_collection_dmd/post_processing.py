"""
after the data collection, carry out the post-processing of the data, update thoes metadata to the database
including 1. cropping images positions
2. beam parameters
"""

from conf import *
import cv2
import json
import pandas as pd
from tqdm import tqdm

# Database Initialization
DB = database.SQLiteDB(DATABASE_ROOT)

# Select the images to process (could based on other fields, adjust the query accordingly)
batch = DB.sql_select("SELECT id, image_path, is_params FROM mmf_dataset_metadata WHERE batch NOT IN (1)") # WHERE batch IN (1, 2, 3, 4, 5)
print("Number of images to process:", len(batch))

# ----------------- Update crop position -----------------
img = cv2.imread(batch['image_path'].iloc[0])
crop_areas = processing.select_crop_areas_center(img, num=2, scale_factor=0.5) 
print("Defined crop areas:", crop_areas)

original_crop = [crop_areas[0]]*len(batch)
fiber_crop = [crop_areas[1]]*len(batch)
df = pd.DataFrame({"id": batch["id"], "original_crop_pos": original_crop, "speckle_crop_pos": fiber_crop})
sql = DB.batch_update("mmf_dataset_metadata", "id", df)
DB.sql_execute(sql, multiple=True)


# ----------------- Update max intensity -----------------
# TODO: calculate and fill the max intensity (pixel value) of the images





# ----------------- Update beam parameters -----------------
print("Calculating beam parameters...")
for i in tqdm(range(len(batch))):
    if batch['is_params'].iloc[i]:
        img = utils.read_narray_image(batch['image_path'].iloc[i])
        cropped_image = img[0:img.shape[0], 0:img.shape[0]]
        params = json.dumps(evaluation.beam_params(cropped_image))
        DB.update_record("mmf_dataset_metadata", "id", batch['id'].iloc[i], "beam_parameters", params)



# ---------------- Information Correction  -----------------
# recallculate the number of images in a batch
sql = """
    SELECT meta.batch AS batch, count(meta.id) AS total_images
    FROM mmf_dataset_metadata AS meta
    LEFT JOIN mmf_experiment_config AS conf
    ON meta.config_id = conf.id
    GROUP BY meta.batch
"""
df = DB.sql_select(sql)
sql = DB.batch_update("mmf_experiment_config", "batch", df)
DB.sql_execute(sql, multiple=True)



# Close the database connection
DB.close()
