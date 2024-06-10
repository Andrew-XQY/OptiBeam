"""
main loop for original-fiber image data collection experiment
"""

from conf import *
import datetime

db = database.SQLiteDB(DATABASE_ROOT)

# set the metadata table name
imageMeta = metadata.ImageMetadata()
confMeta = metadata.ConfigMetaData()

confMeta.set_basic_experiment_metadata("test_experiment", "test_description", "CERN")
confMeta.set_config_metadata({"fiber_config":"1500nm"},{"camera_config":"60000ms"},{"other_config":"15 degree"})

print(confMeta.to_sql_insert("mmf_dataset_config"))


for i in range(50):
    if not db.entry_exists("mmf_dataset_config", "hash", confMeta.get_hash()):
        db.sql_execute(confMeta.to_sql_insert("mmf_dataset_config")) 
    meta = {
            "image_id":f"image_{i}", 
            "capture_time":datetime.datetime.now().strftime('%Y-%m-%d'),
            "original_crop_pos":"(0,0),(100,100)", 
            "speckle_crop_pos":"(0,0),(100,100)", 
            "beam_parameters":"1500nm 570nm no_bending", 
            "num_of_images":3, 
            "metadata_id":db.get_max_id("mmf_dataset_config"),
            "comments":"test comment"
            }
    imageMeta.set_image_metadata(meta)
    db.sql_execute(imageMeta.to_sql_insert("mmf_dataset_metadata")) 
    
db.close()
