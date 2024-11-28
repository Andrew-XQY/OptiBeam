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
# Local image
path_to_images = ["../../ResultsCenter/local_images/MMF/procIMGs/processed",
                  "../../ResultsCenter/local_images/MMF/procIMGs_2/processed"]
paths = utils.get_all_file_paths(path_to_images)
process_funcs = [utils.rgb_to_grayscale, utils.image_normalize, 
                 utils.split_image, lambda x : (x[0] * 255).astype(np.uint8)]

# minst_path = "../../ResultsCenter/local_images/MNIST_ORG/t10k-images.idx3-ubyte"
# imgs_array = read_MNIST_images(minst_path)

# create a queue of image sources
queue.append([simulation.dmd_calibration_pattern_generation()]) 
queue.append([np.ones((256, 256)) * 255]) 
queue.append(simulation.moving_blocks_generator(size=256, block_size=128, intensity=255))
queue.append(simulation.canvas_generator(CANVAS, conf)) 
queue.append(simulation.read_local_generator(paths, process_funcs))


for i in queue:
    for j in i:
        plt.clf()
        plt.imshow(j, cmap='viridis', vmin=0, vmax=255) # cmap='gray' for black and white, and 'viridis' for color
        plt.colorbar()
        plt.draw()  
        plt.pause(2)  # Pause for a short period, allowing the plot to be updated
exit()






































































# import tensorflow as tf
# print(tf.__version__)

# path_to_images = "../../DataWarehouse/MMF/procIMGs/processed"
# paths = utils.get_all_file_paths(path_to_images)
# process_funcs = [utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, lambda x : x[0]]
# loader = utils.ImageLoader(process_funcs)
# imgs_array = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)   
# number_of_images = len(imgs_array)

# print(imgs_array[0].max())





# DB.rename_field("mmf_dataset_metadata", "is_blank", "max_pixel_value")
# DB.retype_field("mmf_dataset_metadata", "max_pixel_value", "TEXT")






# # modify the tables
# sql = """
# UPDATE mmf_experiment_config
# SET image_source = "Georges beam image 1000"
# WHERE id = 1;
# """
# DB.sql_execute(sql)
# DB.close()




# # # Delete images and database according to batch number!!!
# BATCH = 2
# # 
# # select_batch = f"""
# #     SELECT image_path FROM mmf_dataset_metadata WHERE batch = {BATCH};
# # """
# # df = DB.sql_select(select_batch)
# # for image in df['image_path']:
# #     if os.path.exists(image):
# #         os.remove(image)

# tables = ["mmf_dataset_metadata", "mmf_experiment_config"]
# for table in tables:
#     sql = f"""
#         DELETE FROM {table} WHERE batch = {BATCH};
#     """
#     DB.sql_execute(sql)

# DB.close()








# # local images
# load_from_disk = True

# if load_from_disk:
#     path_to_images = "../../DataWarehouse/MMF/procIMGs_2/processed"
#     paths = utils.get_all_file_paths(path_to_images)
#     process_funcs = [utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, lambda x : x[0]]
#     loader = utils.ImageLoader(process_funcs)
#     imgs_array = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)
    
    
# print(imgs_array.shape)
# visualization.plot_narray(imgs_array[0])
