"""
main loop for original-fiber image data collection experiment
"""
from conf import *
from ALP4 import *
import datetime, time
import cv2
import json


def read_MNIST_images(filepath):
    with open(filepath, 'rb') as file:
        # Skip the magic number and read dimensions
        magic_number = int.from_bytes(file.read(4), 'big')  # not used here
        num_images = int.from_bytes(file.read(4), 'big')
        rows = int.from_bytes(file.read(4), 'big')
        cols = int.from_bytes(file.read(4), 'big')

        # Read each image into a numpy array
        images = []
        for _ in range(num_images):
            image = np.frombuffer(file.read(rows * cols), dtype=np.uint8)
            image = image.reshape((rows, cols))
            images.append(image)

        return images
    
    
    
# ------------------- Dataset Parameters ------------------
number_of_images = 1500
is_params = 0
calibration = 1
load_from_disk = False
include_simulation = True  # add the original loaded image into data samples
sim_num = 5    # number of distributions in the simulation
stride = 5  # every stride simulation update steps, load a new image
DMD_DIM = 1024  # DMD final loaded image resolution
DIM = 512    # simulation image resolution
# ---------------------------------------------------------


# DMD Initialization
DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
DMD.display_image(simulation.create_mosaic_image(size=DMD_DIM)) # preload one image for camera calibration

# Cameras Initialization
MANAGER = camera.MultiBaslerCameraManager()
MANAGER.synchronization()

# Database Initialization
DB = database.SQLiteDB(DATABASE_ROOT)
ImageMeta = metadata.ImageMetadata()
ConfMeta = metadata.ConfigMetaData()

# Simulation Initialization (Optional, could just load disk images instead)
CANVAS = simulation.DynamicPatterns(*(DIM, DIM))
CANVAS._distributions = [simulation.GaussianDistribution(CANVAS) for _ in range(sim_num)]


# If load specific images from local disk, set load_from_disk to True
if load_from_disk:
    path_to_images = "../../DataWarehouse/MMF/procIMGs_2/processed"
    paths = utils.get_all_file_paths(path_to_images)
    process_funcs = [utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, lambda x : x[0]]
    loader = utils.ImageLoader(process_funcs)
    imgs_array = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)   
    number_of_images = len(imgs_array)

# minst_path = "../../DataWarehouse/MMF/MNIST_ORG/t10k-images.idx3-ubyte"
# imgs_array = read_MNIST_images(minst_path)


# Setting up the experiment metadata
batch = (DB.get_max("mmf_dataset_metadata", "batch") or 0) + 1  # get the current batch number
experiment_metadata = {
    "experiment_description": "First dataset using DMD, no pertubation included",
    "experiment_location": "DITALab, Cockcroft Institute, UK",
    "experiment_date": datetime.datetime.now().strftime('%Y-%m-%d'),
    "batch": batch,
    "image_source": "beam image" if load_from_disk else "simulation",  # e-beam, proton-beam, simulation, or other dataset like MNIST
    "image_device": "dmd",  # scintillation-screen, dmd, slm, led
    "fiber_config": {
        "fiber_length": "5 m",
        "fiber_name": "1500 micrometer Core-diameter 0.50 NA Step-Index Multimode Fiber Patch Cable",
        "fiber_url": "https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=362&pn=FP1500ERT",
    },
    "camera_config": MANAGER.get_metadata(),
    "other_config": {
        "dmd_config": DMD.get_metadata(),
        "simulation_config": CANVAS.get_metadata() if not load_from_disk else {},
        "light_source": "class 2 laser",
        "temperature": "10 C"
    },
    "purtubations": {},
    "radiation":{}
}

# Start the data collection experiment main loop.
save_dir = DATASET_ROOT + str(batch) # datetime.datetime.now().strftime('%Y-%m-%d')
if not os.path.exists(save_dir): # Create the directory
    os.makedirs(save_dir)
    print(f"Directory '{save_dir}' was created.")
    
ConfMeta.set_metadata(experiment_metadata) 
if not DB.record_exists("mmf_experiment_config", "hash", ConfMeta.get_hash()):
    DB.sql_execute(ConfMeta.to_sql_insert("mmf_experiment_config")) # save the experiment metadata to the database
config_id = DB.get_max("mmf_experiment_config", "id")

count = 0
try:
    for i in range(-1 if calibration else 0, number_of_images):
        if calibration:
            img = simulation.dmd_calibration_pattern_generation()
            
        # select image source
        # ------------------------------ local image --------------------------------
        elif load_from_disk:
            if count >= len(imgs_array): break # local images are already all loaded
            img = imgs_array[i]
        # ---------------------------------------------------------------------------
        
        # ------------------------------- simulation --------------------------------
        else:
            for _ in range(stride):  # update the simulation
                CANVAS.fast_update()
            CANVAS.update()
            img = CANVAS.get_image()
        # ---------------------------------------------------------------------------
        
        img = simulation.pixel_value_remap(img)
        img = simulation.macro_pixel(img, size=int(DMD_DIM/img.shape[0])) 
        origianl = img.copy()
        # Because the DMD is rotated by about 45 degrees, we need to rotate the generated image by ~45 degrees back
        scale = 1 / np.sqrt(2)
        center = (DMD_DIM // 2, DMD_DIM // 2)
        M = cv2.getRotationMatrix2D(center, 47, scale)
        img = cv2.warpAffine(img, M, (DMD_DIM, DMD_DIM), 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(0, 0, 0))
        
        DMD.display_image(img) 
        # time.sleep(0.3)  # If loading speed is too fast, the DMD might has memory error
        
        # capture the image from the cameras (Scheduled action command)
        image = MANAGER.schedule_action_command(int(500 * 1e6)) # schedule for milliseconds later
        if image is not None:
            img_size = (image.shape[0], int(image.shape[1]//2))
            if include_simulation:
                original_image = cv2.resize(origianl, (image.shape[0],image.shape[0])) # add the very original image load on the dmd
                rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
                image = np.hstack((rotated_image, image))
            filename = str(time.time_ns())
            image_path = save_dir + '/' + filename + '.png'
             
            # save the corresponding metadata of the image
            meta = {
                    "image_id":str(time.time_ns()), 
                    "capture_time":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "num_of_images":3 if include_simulation else 2, 
                    "is_params":is_params,
                    "is_calibration":calibration,
                    "image_descriptions":json.dumps({**({"simulation_img": img_size} if include_simulation else {}), 
                                                     "ground_truth_img": img_size, "fiber_output_img": img_size}),
                    "image_path":os.path.abspath(image_path),
                    "config_id":config_id,
                    "batch":batch,
                    "comments":""
                    }
            ImageMeta.set_metadata(meta)
            cv2.imwrite(image_path, image)
            DB.sql_execute(ImageMeta.to_sql_insert("mmf_dataset_metadata")) 
            print(f"Image {i+1} captured.")
            count += 1
            calibration = 0
            
except Exception as e:
    print(f"An error occured, data collection stopped. {e}")


# actual number of images captured
DB.update_record("mmf_experiment_config", "id", config_id, "total_images", count)       
# End the data collection experiment and close everything properly
DB.close()
DMD.end()
MANAGER.end()

