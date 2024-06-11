"""
main loop for original-fiber image data collection experiment
"""
from conf import *
from ALP4 import *
import datetime, time
import cv2

# Cameras Initialization
MANAGER = camera.MultiBaslerCameraManager()
MANAGER.synchronization()

# DMD Initialization
DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
DMD_DIM = 1024

# Database Initialization
DB = database.SQLiteDB(DATABASE_ROOT)
ImageMeta = metadata.ImageMetadata()
ConfMeta = metadata.ConfigMetaData()

# Simulation Initialization (Optional, could just load disk images instead)
DIM = 1024 
CANVAS = simulation.DynamicPatterns(*(DIM, DIM))
CANVAS._distributions = [simulation.GaussianDistribution(CANVAS) for _ in range(10)]

# Setting up the experiment metadata
number_of_images = 50
batch = (DB.get_max("mmf_dataset_metadata", "batch") or 0) + 1  # get the current batch number
experiment_metadata = {
    "experiment_description": "First dataset collection using DMD, no pertubation included. Simulated Stochastic Multiple 2D Guassian Fields used",
    "experiment_location": "DITALab, Cockcroft Institute, UK",
    "experiment_date": datetime.datetime.now().strftime('%Y-%m-%d'),
    "total_images": number_of_images,
    "batch": batch,
    "fiber_config": {
        "fiber_length": "5 m",
        "fiber_name": "1500 µm Core-diameter 0.50 NA Step-Index Multimode Fiber Patch Cable",
        "fiber_url": "https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=362&pn=FP1500ERT",
    },
    "camera_config": MANAGER.get_metadata(),
    "other_config": {
        "dmd_config": DMD.get_metadata(),
        "simulation_config": CANVAS.get_metadata(),
        "light_source": "class 2 laser",
        "temperature": "11 C"
    },
    "purtubations": {},
    "radiation":{}
}
ConfMeta.set_config_metadata(experiment_metadata) 
if not DB.record_exists("mmf_experiment_config", "hash", ConfMeta.get_hash()):
    DB.sql_execute(ConfMeta.to_sql_insert("mmf_experiment_config")) # save the experiment metadata to the database
config_id = DB.get_max("mmf_experiment_config", "id")


# Start the data collection experiment main loop.
stride = 5  # every stride simulation update steps, load a new image
save_dir = DATASET_ROOT + datetime.datetime.now().strftime('%Y-%m-%d')
if not os.path.exists(save_dir): # Create the directory
    os.makedirs(save_dir)
    print(f"Directory '{save_dir}' was created.")

for i in range(number_of_images):
    # ------------------------------ simulation --------------------------------
    for _ in range(stride):  # update the simulation
        CANVAS.fast_update()
    CANVAS.update()
    img = simulation.pixel_value_remap(CANVAS.get_image())
    img = simulation.macro_pixel(img, size=int(DMD_DIM/DIM)) 
    # Because the DMD is rotated by about 45 degrees, we need to rotate the generated image by ~45 degrees back
    scale = 1 / np.sqrt(2)
    center = (DMD_DIM // 2, DMD_DIM // 2)
    M = cv2.getRotationMatrix2D(center, 47, scale)
    img = cv2.warpAffine(img, M, (DMD_DIM, DMD_DIM), 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(0, 0, 0))
    # ---------------------------------------------------------------------------
    
    DMD.display_image(img) 
    time.sleep(0.5)  # If loading speed is too fast, the DMD might has memory error
    
    # capture the image from the cameras (Scheduled action command)
    image = MANAGER.schedule_action_command(int(3000 * 1e6)) # schedule for 3 seconds
    if image is not None:
        original_image = cv2.resize(img, (image.shape[0],image.shape[0])) # add the very original image load on the dmd
        rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
        three_imgs = np.hstack((rotated_image, image))
        filename = str(time.time_ns())
        image_path = save_dir + filename + '.png'
        cv2.imwrite(image_path, three_imgs)
    
        # save the corresponding metadata
        meta = {
                "image_id":str(time.time_ns()), 
                "capture_time":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "num_of_images":3, 
                "image_path":image_path,
                "metadata_id":config_id,
                "batch":batch,
                "comments":""
                }
        ImageMeta.set_image_metadata(meta)
        DB.sql_execute(ImageMeta.to_sql_insert("mmf_dataset_metadata")) 
    else:
        number_of_images -= 1
        
DB.update_record("mmf_experiment_config", "id", config_id, {"total_images":number_of_images})  # actual number of images captured      
    
# End the data collection experiment and close everything properly
DB.close()
DMD.end()
MANAGER.end()

