"""
main loop for original-fiber image data collection experiment
"""

from conf import *
import datetime

db = database.SQLiteDB(DATABASE_ROOT)
# manager = camera.MultiBaslerCameraManager()
# manager.synchronization()


# set the metadata table name
imageMeta = metadata.ImageMetadata()
confMeta = metadata.ConfigMetaData()

confMeta.set_basic_experiment_metadata(experiment_name="Stochastic Multiple 2D Guassian Fields",
                                       experiment_description="First dataset collection using DMD, no pertubation included.",
                                       experiment_location="DITALab, Cockcroft Institute, UK")
confMeta.set_config_metadata(fiber_config={"fiber_length":"5 meters", "fiber_name":"1500 µm Core 0.50 NA Step-Index Multimode Fibers",
                                           "fiber_url":"https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=362&pn=FP1500ERT", 
                                           "purtubations":""},
                             camera_config={"exposure":"0"}, # manager.get_metadata()
                             other_config={"dmd_config":"Texas", "simulation_config":"20 Gaussian", "laser_config":"HeNa-450nm", "temperature":"15 degrees Celsius"})
                             #other_config={"dmd_config":DMD.get_metadata(), "simulation_config":canvas.get_metadata(), "laser_config":"", "temperature":""})
batch = (db.get_max("mmf_dataset_metadata", "batch") or 0) + 1

for i in range(50):
    if not db.entry_exists("mmf_dataset_config", "hash", confMeta.get_hash()):
        db.sql_execute(confMeta.to_sql_insert("mmf_dataset_config")) 
    meta = {
            "image_id":f"image_{i}", 
            "capture_time":datetime.datetime.now().strftime('%Y-%m-%d'),
            "num_of_images":3, 
            "image_path":f"../../ResultsCenter/dataset/image_{i}.png",
            "metadata_id":db.get_max("mmf_dataset_config", "id"),
            "batch":batch,
            "comments":"test comment"
            }
    imageMeta.set_image_metadata(meta)
    db.sql_execute(imageMeta.to_sql_insert("mmf_dataset_metadata")) 
    
    
db.close()
