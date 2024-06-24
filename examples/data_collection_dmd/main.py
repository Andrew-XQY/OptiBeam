"""
main loop for original-fiber image data collection experiment
"""
from conf import *
from ALP4 import *
import datetime, time
import cv2
import json
    
    
# --------------------- Dataset Parameters --------------------
number_of_images = 60  # for simulation, this is the number of images to generate in this batch
is_params = 0  # if the image contains beam parameters (simulation and MNIST don't)
calibration = 1  # if include a calibration image (first one in the batch)
load_from_disk = False  # load images from local disk instead of running simulation
include_simulation = True  # add the original loaded image into data samples
DMD_DIM = 1024  # DMD final loaded image resolution
# -------------------------------------------------------------



# ------------------- Hardware Initialization ------------------
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

# Simulation Initialization (Optional, could just load disk images or any image list instead)
sim_num = 100    # number of distributions in the simulation
fade_rate = 0.96  # with 100 sim_num. around 0.96 looks good
min_std=0.05 
max_std=0.1
max_intensity=100
dim = 512   # simulation image resolution  
CANVAS = simulation.DynamicPatterns(dim, dim)
CANVAS._distributions = [simulation.StaticGaussianDistribution(CANVAS) for _ in range(sim_num)] 
# CANVAS._distributions = [simulation.GaussianDistribution(CANVAS) for _ in range(sim_num)] 
# -------------------------------------------------------------


# If load specific images from local disk, set load_from_disk to True
if load_from_disk:
    path_to_images = "../../DataWarehouse/MMF/procIMGs_2/processed"
    paths = utils.get_all_file_paths(path_to_images)
    process_funcs = [utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, lambda x : x[0]]
    loader = utils.ImageLoader(process_funcs)
    imgs_array = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)   
    number_of_images = len(imgs_array)

# minst_path = "../../DataWarehouse/MNIST_ORG/t10k-images.idx3-ubyte"
# imgs_array = read_MNIST_images(minst_path)


# Setting up the experiment metadata
batch = (DB.get_max("mmf_dataset_metadata", "batch") or 0) + 1  # get the current batch number
experiment_metadata = {
    "experiment_description": "Second dataset using DMD, muit-gaussian distributions",
    "experiment_location": "DITALab, Cockcroft Institute, UK",
    "experiment_date": datetime.datetime.now().strftime('%Y-%m-%d'),
    "batch": batch,
    "image_source": "beam image" if load_from_disk else "simulation",  # e-beam, proton-beam, simulation, or other dataset like MNIST
    "image_device": "dmd",  # scintillation-screen, dmd, slm, led, OTR.
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
        "temperature": ""
    },
    "purtubations": None,
    "radiation":None
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


intensity_sum = 0
comment = None
try:
    for count in range(-1 if calibration else 0, number_of_images):
        if calibration:
            img = simulation.dmd_calibration_pattern_generation()
            
        # select image source
        # ------------------------------ local image --------------------------------
        elif load_from_disk:
            if count >= len(imgs_array): break # local images are already all loaded
            img = imgs_array[count]
        # ---------------------------------------------------------------------------
        
        # -------------------------- simulation (dynamic) ---------------------------
        # else:
        #     for _ in range(stride):  # update the simulation
        #         CANVAS.fast_update()
        #     CANVAS.update()
        #     CANVAS.thresholding(1)    
        #     img = CANVAS.get_image()
        # ---------------------------------------------------------------------------
        
        # ------------------------------- simulation --------------------------------
        # else:
        #     CANVAS.update(min_std=min_std, max_std=max_std, max_intensity=max_intensity, fade_rate=fade_rate)  # around 0.95 looks good
        #     CANVAS.thresholding(1)
        #     img = CANVAS.get_image()
        # ---------------------------------------------------------------------------
        
        # ----------------------------- special experiment --------------------------
        # else:  # Intensity correction
        #     if intensity_sum == 0 or intensity_sum >= 150:
        #         while True:
        #             CANVAS.update(min_std=min_std, max_std=max_std, max_intensity=max_intensity, fade_rate=fade_rate)
        #             CANVAS.thresholding(1)
        #             img = CANVAS.get_image()
        #             if img.max() + 150 <= 255 and not CANVAS.is_blank():
        #                 break
        #         intensity_sum = 0
        #     else:
        #         img = np.where(img > 0, np.clip(img+10, 0, 255), 0)
        #     comment = f"intensity test: +{intensity_sum}"  # superposition test, intensity test
        #     intensity_sum += 10
        
        
        # else:  # superposition experiment (assum only two Gaussian distributions in the simulation)
        #     sim_num = 4
        #     fade_rate = 0
        #     group = count % (sim_num+1)
        #     sub_batch = count // (sim_num+1)
        #     if group == 0:
        #         CANVAS._distributions = [simulation.StaticGaussianDistribution(CANVAS) for _ in range(sim_num)] 
        #         CANVAS.update(min_std=min_std, max_std=max_std, max_intensity=max_intensity, fade_rate=fade_rate)
        #         CANVAS.thresholding(1)
        #     else:
        #         CANVAS.clear_canvas()
        #         CANVAS.apply_specific_distribution(group-1)
        #     img = CANVAS.get_image()
        #     comment = f"superposition test: batch |{sub_batch}|{group}"
        # ---------------------------------------------------------------------------
        
        
        # Preprocess the image before displaying on the DMD
        # img = simulation.pixel_value_remap(img)   # remap the intensity will decrease the diversity of the images
        display = img.copy()
        display = simulation.macro_pixel(display, size=int(DMD_DIM/display.shape[0])) 
        
        # Because the DMD is rotated by about 45 degrees, we need to rotate the generated image by ~45 degrees back
        scale = 1 / np.sqrt(2)
        center = (DMD_DIM // 2, DMD_DIM // 2)
        M = cv2.getRotationMatrix2D(center, 47, scale)  # 47 is the angle to rotate to the right orientation in this case
        display = cv2.warpAffine(display, M, (DMD_DIM, DMD_DIM), 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(0, 0, 0))

        DMD.display_image(display)  # if loading too fast, the DMD might report memory error
        
        # capture the image from the cameras (Scheduled action command)
        image = MANAGER.schedule_action_command(int(500 * 1e6)) # schedule for milliseconds later
        if image is not None:
            img_size = (image.shape[0], int(image.shape[1]//2))
            if include_simulation:
                original_image = cv2.resize(img, (image.shape[0],image.shape[0])) # add the very original image load on the dmd
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
                    "is_blank":1 if CANVAS.is_blank() else 0,
                    "max_pixel_value":img.max(),
                    "image_descriptions":json.dumps({**({"simulation_img": img_size} if include_simulation else {}), 
                                                     "ground_truth_img": img_size, "fiber_output_img": img_size}),
                    "image_path":os.path.abspath(image_path),
                    "config_id":config_id,
                    "batch":batch,
                    "comments":comment
                    }
            ImageMeta.set_metadata(meta)
            cv2.imwrite(image_path, image)
            DB.sql_execute(ImageMeta.to_sql_insert("mmf_dataset_metadata")) 
            print(f"Image {count+1} captured.")
            count += 1
            calibration = 0
            
except Exception as e:
    print(f"An error occured, data collection stopped. {e}")


# actual number of images captured
DB.update_record("mmf_experiment_config", "id", config_id, "total_images", count+1)       
# End the data collection experiment and close everything properly
DB.close()
DMD.end()
MANAGER.end()

