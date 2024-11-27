"""
    This script is the main script for fiber image/speckle pattern data collection
"""
from conf import *
from ALP4 import *
import datetime, time
import cv2
import json


# ============================
# Dataset Parameters
# ============================
conf = {
    'number_of_images': 5,  # simulation: number of images to generate in this batch
    'is_params': False,  # if the image contains beam parameters (MNIST don't)
    'calibration': True,  # if include a calibration image (first one in the batch)
    'load_from_disk': False,  # if load images from local disk
    'include_simulation': False,  # if include the original simulation image loaded on the DMD
    'dmd_dim': 1024,  # DMD working square area resolution
    'dmd_rotation': 47+90,  # DMD rotation angle for image orientation correction
    'crop_areas': [((872, 432), (1032, 592)), ((2817, 437), (3023, 643))],  # crop areas for the camera images
    'sim_pattern_max_num': 100,  # simulation: maximum number of distributions in the simulation
    'sim_fade_rate': 0.96,  # simulation: the probability of a distribution to disappear
    'sim_std_1': 0.03, # simulation: lower indication of std
    'sim_std_2': 0.2, # simulation: higher indication of std
    'sim_max_intensity': 100, # simulation: peak pixel intensity in a single distribution
    'sim_dim': 512,   # simulation: simulated image resolution
}


# ============================
# Hardware/Software Initialization
# ============================
# Database Initialization
DB = database.SQLiteDB(DATABASE_ROOT)
ImageMeta = metadata.ImageMetadata()
ConfMeta = metadata.ConfigMetaData()
# DMD Initialization 
DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
calibration_img = simulation.generate_radial_gradient() # generate_upward_arrow(), dmd_calibration_pattern_generation() np.ones((256, 256)) * 255
calibration_img = simulation.macro_pixel(calibration_img, size=int(conf['dmd_dim']/calibration_img.shape[0])) 
DMD.display_image(dmd.dmd_img_adjustment(calibration_img, conf['dmd_dim'], angle=conf['dmd_rotation'])) # preload one image for camera calibration
# Cameras Initialization
MANAGER = camera.MultiBaslerCameraManager()
MANAGER.initialize()
MANAGER.synchronization()


# ============================
# Select crop areas (optional)
# ============================
# take a sample image to (later manually) select crop areas for automatic resizing
# test_img = MANAGER.schedule_action_command(int(300 * 1e6)) # schedule for milliseconds later
# crop_areas = processing.select_crop_areas_center(test_img, num=2, scale_factor=0.4) 
# print("Crop areas selected: ", crop_areas)
# exit()


# ============================
# Image sources initialization
# ============================
experiments = [ # Define all experiments to be conducted
    {'name': 'calibration', 'len': None},
    {'name': 'local_intensity_coupling', 'len': None},
    {'name': 'full_screen_intensity', 'len': None},
    {'name': 'testset', 'len': None},
    {'name': 'training', 'len': None}
    ]
queue = []

# Simulation Initialization
CANVAS = simulation.DynamicPatterns(conf['sim_dim'], conf['sim_dim'])
CANVAS._distributions = [simulation.StaticGaussianDistribution(CANVAS) for _ in range(conf['sim_pattern_max_num'])] 

def simulation():
    CANVAS.update(std_1=conf['sim_std_1'], std_2=conf['sim_std_2'],
                max_intensity=conf['sim_max_intensity'], fade_rate=conf['sim_fade_rate'],
                distribution='normal') 
    #CANVAS.thresholding(1)
    img = CANVAS.get_image()
    comment = {'num_of_distributions': CANVAS.num_of_distributions(), 
                'distributions_metadata': CANVAS.get_distributions_metadata()}
    return img, comment



queue.append(simulation.dmd_calibration_pattern_generation()) 
queue.append(simulation.moving_blocks_generator(size=256, block_size=32, intensity=255))
queue.append(simulation.image_generator_wrapper(simulation(), conf['number_of_images'])) 







# If load specific images from local disk, set load_from_disk to True
if conf['load_from_disk']:
    path_to_images = ["../../ResultsCenter/local_images/MMF/procIMGs/processed",
                      "../../ResultsCenter/local_images/MMF/procIMGs_2/processed"]
    paths = utils.get_all_file_paths(path_to_images)
    process_funcs = [utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, lambda x : x[0]]
    loader = utils.ImageLoader(process_funcs)
    imgs_array = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)   

img = (img * 255).astype(np.uint8) # convert 0-1 to 0-255, only apply to certain images

# minst_path = "../../ResultsCenter/local_images/MNIST_ORG/t10k-images.idx3-ubyte"
# imgs_array = read_MNIST_images(minst_path)



# ============================
# Setting up experiment metadata
# ============================
batch = (DB.get_max("mmf_dataset_metadata", "batch") or 0) + 1  # get the current batch number
experiment_metadata = {
    "experiment_description": "DMD-training-500", # Second dataset using DMD, muit-gaussian distributions, small scale
    "experiment_location": "DITALab, Cockcroft Institute, UK",
    "experiment_date": datetime.datetime.now().strftime('%Y-%m-%d'),
    "batch": batch,
    "image_source": "beam image" if conf['load_from_disk'] else "simulation",  # e-beam, proton-beam, simulation, or other dataset like MNIST
    "image_device": "dmd",  # scintillation-screen, dmd, slm, led, OTR.
    "fiber_config": {
        "fiber_length": "5 m",
        "fiber_name": "600 micrometer Core-diameter Step-Index Multimode Fiber Patch Cable",
        "fiber_url": "",
    },
    "camera_config": MANAGER.get_metadata(),
    "other_config": {
        "dmd_config": DMD.get_metadata(),
        "simulation_config": CANVAS.get_metadata() if not conf['load_from_disk'] else None,
        "light_source": "CPS532-C2",
        'other_notes': (  # if simulation, save the simulation parameters
            f"sim_pattern_max_num={conf['sim_pattern_max_num']}; "
            f"sim_fade_rate={conf['sim_fade_rate']}; "
            f"sim_std_1={conf['sim_std_1']}; "
            f"sim_std_2={conf['sim_std_2']}; "
            f"sim_max_intensity={conf['sim_max_intensity']}; "
            f"sim_dim={conf['sim_dim']}"
        )},  # Additional simulation parameters
    "purtubations": None,
    "radiation" : None
}

save_dir = DATASET_ROOT + str(batch) # datetime.datetime.now().strftime('%Y-%m-%d')
if not os.path.exists(save_dir): # create sub batch folder
    os.makedirs(save_dir)
    print(f"Directory '{save_dir}' was created.")
    
ConfMeta.set_metadata(experiment_metadata) 
if not DB.record_exists("mmf_experiment_config", "hash", ConfMeta.get_hash()):
    DB.sql_execute(ConfMeta.to_sql_insert("mmf_experiment_config")) # save the experiment metadata to the database
else:
    raise ValueError("The experiment metadata already exists in the database.")
config_id = DB.get_max("mmf_experiment_config", "id")


# ============================
# data collection pipeline
# ============================
# Start the data collection experiment main loop.
try:
    # update experiment metadata
    experiment_metadata[''] = None
    
except Exception as e:
    print(f"An error occured, data collection stopped. {e}")




try:
    for count in range(-1 if conf['calibration'] else 0, conf['number_of_images']):
        comment = None
        if conf['calibration']:
            img = simulation.dmd_calibration_pattern_generation()
            
        # go through image source
        
        
        # Preprocess the image before displaying on the DMD
        display = img.copy()
        display = simulation.macro_pixel(display, size=int(conf['dmd_dim']/display.shape[0])) 
        # Because the DMD is rotated by about 45 degrees, we need to rotate the generated image by ~45 degrees back
        display = dmd.dmd_img_adjustment(display, conf['dmd_dim'], angle=conf['dmd_rotation'])
        DMD.display_image(display)  # if loading too fast, the DMD might report memory error
        
        # capture the image from the cameras (Scheduled action command)
        image = MANAGER.schedule_action_command(int(200 * 1e6)) # schedule for milliseconds later
        if image is not None:
            img_size = (image.shape[0], int(image.shape[1]//2))  
            if conf['include_simulation']:
                original_image = cv2.resize(img, (image.shape[0],image.shape[0])) # add the very original image load on the dmd
                image = np.hstack((original_image, image))
            filename = str(time.time_ns()) + '.png'
            image_path = save_dir + '/' + filename # absolute path save on the local machine
            relative_path = '/'.join(['datasets', str(batch), filename]) # changed to relative path to dataset root instead of absolute path on machine
            
            # update and save the corresponding metadata of the image
            meta = {
                    "image_id":str(time.time_ns()), 
                    "capture_time":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "num_of_images":3 if conf['include_simulation'] else 2, 
                    "is_params":conf['is_params'],
                    "is_calibration":conf['calibration'],
                    "max_pixel_value":img.max(),
                    "image_descriptions":json.dumps({**({"simulation_img": img_size} if conf['include_simulation'] else {}), 
                                                     "ground_truth_img": img_size, "fiber_output_img": img_size}),
                    "image_path":relative_path,  
                    "config_id":config_id,
                    "batch":batch,
                    "comments":comment
                    }
            ImageMeta.set_metadata(meta)
            # final resize and save the image
            image = processing.crop_image_from_coordinates(image, conf['crop_areas'])
            cv2.imwrite(image_path, image)
            DB.sql_execute(ImageMeta.to_sql_insert("mmf_dataset_metadata")) 
            print(f"Image {count+1} captured.")
            count += 1
            conf['calibration'] = 0
        DMD.free_memory()
            
except Exception as e:
    print(f"An error occured, data collection stopped. {e}")
    
# update actual number of images captured
DB.update_record("mmf_experiment_config", "id", config_id, "total_images", count+1)      


# ============================
# Close everything properly
# ============================
DB.close()  
DMD.end()
MANAGER.end()

