"""
main loop for original-fiber image data collection experiment
"""
from conf import *
from ALP4 import *
import datetime, time
import cv2
import json
    
    
# --------------------- Dataset Parameters --------------------

number_of_images = 2400 # for simulation, this is the number of images to generate in this batch
is_params = 0  # if the image contains beam parameters (simulation and MNIST don't)
calibration = 1  # if include a calibration image (first one in the batch)
load_from_disk = False  # load images from local disk instead of running simulation
include_simulation = False  # add the original loaded image into data samples
DMD_DIM = 1024  # DMD final loaded image resolution
# -------------------------------------------------------------




# ------------------- Hardware Initialization ------------------
# DMD Initialization
DMD_ROTATION = 47+90  # DMD rotation angle
DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
# calibration_img = np.ones((256, 256)) * 255
# calibration_img = simulation.dmd_calibration_corner_dots(size = 256, dot_size= 5)
# calibration_img = simulation.dmd_calibration_center_dot(size = 256, dot_size= 64) 
# calibration_img = simulation.dmd_calibration_pattern_generation()
# calibration_img = simulation.generate_upward_arrow()
# calibration_img = simulation.generate_solid_circle()
calibration_img = simulation.generate_radial_gradient()
calibration_img = simulation.macro_pixel(calibration_img, size=int(DMD_DIM/calibration_img.shape[0])) 
DMD.display_image(dmd.dmd_img_adjustment(calibration_img, DMD_DIM, angle=DMD_ROTATION)) # preload one image for camera calibration

# Cameras Initialization
MANAGER = camera.MultiBaslerCameraManager()
MANAGER.initialize()
MANAGER.synchronization()

# take a sample image to select crop areas for later resizing
# calibration_img = simulation.dmd_calibration_pattern_generation()
# calibration_img = simulation.macro_pixel(calibration_img, size=int(DMD_DIM/calibration_img.shape[0]))
# DMD.display_image(dmd.dmd_img_adjustment(calibration_img, DMD_DIM, angle=DMD_ROTATION))
# test_img = MANAGER.schedule_action_command(int(300 * 1e6)) # schedule for milliseconds later
# crop_areas = processing.select_crop_areas_center(test_img, num=2, scale_factor=0.4) 
# print("Crop areas selected: ", crop_areas)
crop_areas = [((870, 432), (1030, 592)), ((2315, 57), (3385, 1127))]  # manually set the crop areas


# Database Initialization
DB = database.SQLiteDB(DATABASE_ROOT)
ImageMeta = metadata.ImageMetadata()
ConfMeta = metadata.ConfigMetaData()

# Simulation Initialization (Optional, could just load disk images or any image list instead)
sim_num = 100    # number of distributions in the simulation
fade_rate = 0.96  # with 100 sim_num. around 0.96 looks good
std_1=0.03 
std_2=0.2
# std_1 = 0.15
# std_2 = 0.12
max_intensity=100
dim = 512   # simulation image resolution 

stride = 5  # number of simulation updates per image, only for dynamic simulation
CANVAS = simulation.DynamicPatterns(dim, dim)
CANVAS._distributions = [simulation.StaticGaussianDistribution(CANVAS) for _ in range(sim_num)] 
# CANVAS._distributions = [simulation.GaussianDistribution(CANVAS) for _ in range(sim_num)] 
# -------------------------------------------------------------


# If load specific images from local disk, set load_from_disk to True
if load_from_disk:
    path_to_images = ["../../DataWarehouse/MMF/procIMGs/processed",
                      "../../DataWarehouse/MMF/procIMGs_2/processed"]
    paths = utils.get_all_file_paths(path_to_images)
    process_funcs = [utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, lambda x : x[0]]
    loader = utils.ImageLoader(process_funcs)
    imgs_array = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)   

# minst_path = "../../DataWarehouse/MNIST_ORG/t10k-images.idx3-ubyte"
# imgs_array = read_MNIST_images(minst_path)


# ------------------- Define Image Generator Here ------------------
# another option is to create a image generator
image_generator = simulation.position_intensity_generator()
# image_generator = None



# Setting up the experiment metadata
batch = (DB.get_max("mmf_dataset_metadata", "batch") or 0) + 1  # get the current batch number
experiment_metadata = {
    "experiment_description": "Superposition-7", # Second dataset using DMD, muit-gaussian distributions, small scale
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
        "simulation_config": CANVAS.get_metadata() if not load_from_disk else None,
        "light_source": "class 2 laser",
        "other_notes": f"sim_num={sim_num}; fade_rate={fade_rate}; min_std={std_1}; max_std={std_2}; max_intensity={max_intensity}; dim={dim}"},  # if simulation
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
else:
    raise ValueError("The experiment metadata already exists in the database.")
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
            img = (img * 255).astype(np.uint8) # convert 0-1 to 0-255, only apply to certain images
        # ---------------------------------------------------------------------------
        
        # ------------------------------- simulation --------------------------------
        # else:
        #     CANVAS.update(std_1=std_1, std_2=std_2,
        #                   max_intensity=max_intensity, fade_rate=fade_rate,
        #                   distribution='normal') 
        #     #CANVAS.thresholding(1)
        #     img = CANVAS.get_image()
        #     comment = CANVAS.num_of_distributions()
        # ---------------------------------------------------------------------------
        
        # -------------------------------- generator --------------------------------
        # else:  
        #     img, sample_info = next(image_generator)
        #     comment = sample_info
        # else:  
        #     img = next(image_generator)
        # ---------------------------------------------------------------------------
        
        
        # ----------------------------- special experiment --------------------------
        # else:  # Intensity correction
        #     num_in_group = 15
        #     group_no = count // num_in_group
        #     group_index = count % num_in_group
        #     increment = 10
        #     if group_index == 0 or group_index == num_in_group:
        #         while True:
        #             CANVAS.update(min_std=min_std, max_std=max_std, max_intensity=max_intensity, fade_rate=fade_rate)
        #             CANVAS.thresholding(1)
        #             img = CANVAS.get_image()
        #             if img.max() + num_in_group*increment <= 255 and not CANVAS.is_blank():
        #                 break
        #         intensity_sum = 0
        #     else:
        #         img = np.where(img > 0, np.clip(img+intensity_sum, 0, 255), 0)
        #     comment = {"item": "intensity_test", 
        #                "group": group_no, "intensity_increment":intensity_sum}
        #     intensity_sum += 10
        
        
        # else: # time shift experiment, input the same image observe over a long period of time
        #     img = simulation.generate_radial_gradient()
        #     pause_time = 50
        #     time.sleep(pause_time)
        #     comment = {"item": "time_shift_test", "time": pause_time * count}
        
        
        else:  # superposition experiment (assum fixed Gaussian distributions in the simulation)
            sim_num = 7 
            fade_rate = 0
            max_intensity = 120
            group_no = count % (sim_num+1)
            sub_batch = count // (sim_num+1)
            if group_no == 0:
                CANVAS._distributions = [simulation.StaticGaussianDistribution(CANVAS) for _ in range(sim_num)] 
                CANVAS.update(std_1=std_1, std_2=std_2,
                              max_intensity=max_intensity, fade_rate=fade_rate)
                CANVAS.thresholding(1)
            else:
                CANVAS.clear_canvas()
                CANVAS.apply_specific_distribution(group_no-1)
            img = CANVAS.get_image()
            comment = {"item": f"superposition_test_{sim_num}", 
                       "group": sub_batch, "distribution_index":group_no}
        # ---------------------------------------------------------------------------
        
        
        # Preprocess the image before displaying on the DMD
        display = img.copy()
        display = simulation.macro_pixel(display, size=int(DMD_DIM/display.shape[0])) 
        # Because the DMD is rotated by about 45 degrees, we need to rotate the generated image by ~45 degrees back
        display = dmd.dmd_img_adjustment(display, DMD_DIM, angle=DMD_ROTATION)
        DMD.display_image(display)  # if loading too fast, the DMD might report memory error
        
        # capture the image from the cameras (Scheduled action command)
        image = MANAGER.schedule_action_command(int(200 * 1e6)) # schedule for milliseconds later
        if image is not None:
            img_size = (image.shape[0], int(image.shape[1]//2))  
            if include_simulation:
                original_image = cv2.resize(img, (image.shape[0],image.shape[0])) # add the very original image load on the dmd
                image = np.hstack((original_image, image))
            filename = str(time.time_ns())
            image_path = save_dir + '/' + filename + '.png'
             
            # save the corresponding metadata of the image
            meta = {
                    "image_id":str(time.time_ns()), 
                    "capture_time":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "num_of_images":3 if include_simulation else 2, 
                    "is_params":is_params,
                    "is_calibration":calibration,
                    "max_pixel_value":img.max(),
                    "image_descriptions":json.dumps({**({"simulation_img": img_size} if include_simulation else {}), 
                                                     "ground_truth_img": img_size, "fiber_output_img": img_size}),
                    "image_path":os.path.abspath(image_path),
                    "config_id":config_id,
                    "batch":batch,
                    "comments":comment
                    }
            ImageMeta.set_metadata(meta)
            # final resize and save the image
            image = processing.crop_image_from_coordinates(image, crop_areas)
            cv2.imwrite(image_path, image)
            DB.sql_execute(ImageMeta.to_sql_insert("mmf_dataset_metadata")) 
            print(f"Image {count+1} captured.")
            count += 1
            calibration = 0
            DMD.free_memory()
            
except Exception as e:
    print(f"An error occured, data collection stopped. {e}")


# actual number of images captured
DB.update_record("mmf_experiment_config", "id", config_id, "total_images", count+1)       
# End the data collection experiment and close everything properly
DB.close()
DMD.end()
MANAGER.end()

