"""
    This script is the main script for fiber image/speckle pattern data collection
"""
from conf import *
from ALP4 import *
from tqdm import tqdm
import datetime, time
import cv2, json
import subprocess
import traceback


# ============================
# Dataset Parameters
# ============================
conf = {
    'config_crop_area': True, 
    'camera_order_flip': True,  # camera order flip
    'cam_schedule_time': int(300 * 1e6),  # camera schedule time in milliseconds
    'base_resolution': (256, 256),  # base resolution for all images
    'number_of_images': 50000,  # simulation: number of images to generate in this batch
    'number_of_test': None,  # left none for all images
    'number_of_minst': 100,
    'temporal_shift_freq': 50,  # simulation: temporal shift frequency   
    'dmd_dim': 1024,  # DMD working square area resolution
    'dmd_rotation': 47+90,  # DMD rotation angle for image orientation correction
    'dmd_bitDepth': 8,  # DMD bit depth
    'dmd_picture_time': 20000,  # DMD picture time in microseconds, corresponds to 50 Hz -> 20000, 10 Hz -> 100000
    'dmd_alp_version': '4.3',  # DMD ALP version
    'crop_areas': [((865, 425), (1031, 591)), ((2757, 342), (3217, 802))],  # crop areas for the camera images
    'sim_pattern_max_num': 100,  # simulation: maximum number of distributions in the simulation
    'sim_fade_rate': 0.96,  # simulation: the probability of a distribution to disappear
    'sim_std_1': 0.03, # simulation: lower indication of std
    'sim_std_2': 0.2, # simulation: higher indication of std
    'sim_max_intensity': 100, # simulation: peak pixel intensity in a single distribution
    'sim_dim': 512,   # simulation: simulated image resolution
}

# [((871, 432), (1031, 592)), ((2867, 446), (3059, 638))]

# ============================
# Hardware/Software Initialization
# ============================
# DMD Initialization 
DMD = dmd.ViALUXDMD(ALP4(version = conf['dmd_alp_version'])) 
DMD.set_pictureTime(conf['dmd_picture_time'])
# generate_upward_arrow(), dmd_calibration_pattern_generation()   generate_circle_fiber_coupling_pattern(line_width=20)
calibration_img = np.ones((256, 256)) * 100
# calibration_img = simulation.generate_radial_gradient()
calibration_img = simulation.macro_pixel(calibration_img, size=int(conf['dmd_dim']/calibration_img.shape[0])) 
DMD.display_image(dmd.dmd_img_adjustment(calibration_img, conf['dmd_dim'], angle=conf['dmd_rotation'])) # preload for calibration
# Cameras Initialization
MANAGER = camera.MultiBaslerCameraManager()
if conf['camera_order_flip']: MANAGER.flip = True
MANAGER.initialize()
MANAGER.synchronization()


# ============================
# Select crop areas (optional steps), process will stop.
# ============================
if conf['config_crop_area']:
    # take a sample image to (later manually) select crop areas for automatic resizing
    calibration_img = simulation.dmd_calibration_pattern_generation()
    calibration_img = simulation.macro_pixel(calibration_img, size=int(conf['dmd_dim']/calibration_img.shape[0])) 
    DMD.display_image(dmd.dmd_img_adjustment(calibration_img, conf['dmd_dim'], angle=conf['dmd_rotation'])) # preload for calibration
    test_img = MANAGER.schedule_action_command(conf['cam_schedule_time']) # schedule for milliseconds later
    test_img = processing.add_grid(test_img, partitions=50)
    crop_areas = processing.select_crop_areas_corner(test_img, num=2, scale_factor=0.4) 
    sys.exit(f"Crop areas selected: {crop_areas} \nProcedure completed.")


# ============================
# Image sources queue initialization
# ============================
DB = database.SQLiteDB(DATABASE_ROOT)
ImageMeta = metadata.ImageMetadata()
ConfMeta = metadata.ConfigMetaData()

# Simulation Initialization
CANVAS = simulation.DynamicPatterns(conf['sim_dim'], conf['sim_dim'])
CANVAS._distributions = [simulation.StaticGaussianDistribution(CANVAS) for _ in range(conf['sim_pattern_max_num'])] 
# Local image
paths = utils.get_all_file_paths(path_to_images) 
if conf['number_of_test']:
    paths = paths[:conf['number_of_test']]
process_funcs = [utils.rgb_to_grayscale, utils.split_image, lambda x : x[0].astype(np.uint8)]

if conf['number_of_minst']:
    imgs_array = read_MNIST_images(minst_path)[:conf['number_of_minst']]
    minst_len = len(imgs_array)
    imgs_array = utils.list_to_generator(imgs_array)
    print(type(imgs_array))


# create a queue of image sources
# simulation_config, other_notes, experiment_description, image_source, purpose, images_per_sample, is_params, is_calibration
queue = []
queue.append({'experiment_description':'calibration image', 
              'purpose':'calibration',
              'image_source':'simulation',
              'images_per_sample':2,
              'is_calibration':True,
              'data':[simulation.dmd_calibration_pattern_generation()],
              'len':1}) 
queue.append({'experiment_description': 'full screen image',
              'purpose':'intensity_full',
              'image_source':'simulation',
              'images_per_sample':2,
              'data': [np.ones(conf['base_resolution']) * 100],
              'len':1}) 
queue.append({'experiment_description':'position based coupling intensity',
              'purpose':'intensity_position',
              'image_source':'simulation',
              'images_per_sample':2,
              'data':simulation.moving_blocks_generator(size=conf['base_resolution'][0], block_size=32, intensity=255),
              'len':64}) 
queue.append({'experiment_description':'2d multi-gaussian distributions simulation',
              'purpose':'training',
              'image_source':'simulation',
              'images_per_sample':2,
              'simulation_config':CANVAS.get_metadata(),
              'other_notes':{key: value for key, value in conf.items() if 'sim' in key},
              'data':simulation.temporal_shift(conf['temporal_shift_freq'])(simulation.canvas_generator)(CANVAS, conf),
              'len':conf['number_of_images'] + utils.ceil_int_div(conf['number_of_images'], conf['temporal_shift_freq'])}) 
queue.append({'experiment_description':'local real beam image for evaluation',
              'purpose':'testing',
              'images_per_sample':2,
              'image_source':'e-beam',
              'is_params':True,
              'data':simulation.temporal_shift(conf['temporal_shift_freq'])(simulation.read_local_generator)(paths, process_funcs),
              'len':len(paths) + utils.ceil_int_div(len(paths), conf['temporal_shift_freq'])}) 
queue.append({'experiment_description':'MINST for fun',
              'purpose':'fun',
              'images_per_sample':2,
              'image_source':'MINST',
              'data':simulation.temporal_shift(conf['temporal_shift_freq'])(utils.identity)(imgs_array),
              'len':minst_len + utils.ceil_int_div(minst_len, conf['temporal_shift_freq'])}) 


# ============================
# data collection pipeline
# ============================
try:
    for experiment in queue:
        # Setting up experiment metadata
        batch = (DB.get_max("mmf_dataset_metadata", "batch") or 0) + 1  # get the current batch number
        experiment_metadata = {
            "experiment_location": "DITALab, Cockcroft Institute, UK",
            "experiment_date": datetime.datetime.now().strftime('%Y-%m-%d'),
            "image_device": "dmd",  # scintillation-screen, dmd, slm, led, OTR.
            "fiber_config": {
                "fiber_length": "10 m",
                "fiber_name": "600 micrometer Core-diameter Step-Index Multimode Fiber Patch Cable",
                "fiber_url": "FT600UMT",
            },
            "camera_config": MANAGER.get_metadata(), # assume not changing during entire experiment
            "other_config": {
                "dmd_config": DMD.get_metadata(), 
                "simulation_config": experiment.get("simulation_config", None),
                "light_source": "CPS532-C2",
                'other_notes': experiment.get("other_notes", None)},  # Additional simulation parameters
            "purtubations": experiment.get("purtubations", None),
            "radiation" : experiment.get("radiation", None),
            "batch": batch,
            "experiment_description": experiment.get("experiment_description", None), 
            "image_source": experiment.get("image_source", None)  # e-beam, proton-beam, simulation, MNIST
        }

        save_dir = DATASET_ROOT + str(batch) # datetime.datetime.now().strftime('%Y-%m-%d')
        if not os.path.exists(save_dir): # create sub batch folder
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' was created.")
            
        ConfMeta.set_metadata(experiment_metadata) 
        if not DB.record_exists("mmf_experiment_config", "hash", ConfMeta.get_hash()):
            DB.sql_execute(ConfMeta.to_sql_insert("mmf_experiment_config")) # save the experiment metadata to the database
        else:
            print("Notice: The experiment metadata already exists in the database.")
            #raise ValueError("The experiment metadata already exists in the database.")
        config_id = DB.get_max("mmf_experiment_config", "id")

        # print to indicate the current experiment name
        print(f"---> Starting experiment: {experiment['experiment_description']}")
        for img in tqdm(experiment['data'], total=experiment['len']):
            comment = None
            if isinstance(img, tuple): # check if the generator returns a tuple (image, comment)
                img, comment = img
            display = img.copy()
            display = simulation.macro_pixel(display, size=int(conf['dmd_dim']/display.shape[0])) 
            display = dmd.dmd_img_adjustment(display, conf['dmd_dim'], angle=conf['dmd_rotation'])
            DMD.display_image(display)  # if loading too fast, the DMD might report memory error
            
            # capture the image from the cameras (Scheduled action command)
            image = MANAGER.schedule_action_command(conf['cam_schedule_time']) 
            if image is not None:
                img_size = (image.shape[0], int(image.shape[1]//2))  
                if experiment.get("include_simulation", False): # optional, add the very original image
                    original_image = cv2.resize(img, (image.shape[0],image.shape[0])) 
                    image = np.hstack((original_image, image))
                filename = str(time.time_ns()) + '.png'
                image_path = save_dir + '/' + filename # absolute path save on the local machine
                relative_path = '/'.join(['dataset', str(batch), filename]) # relative path save in the database
                # crop the image to regions of interest
                image = processing.crop_image_from_coordinates(image, conf['crop_areas'])
                # image statistics info
                ground_truth, speckle = utils.split_image(image)
                stats = {'ground_truth_img':analysis.analyze_image(ground_truth),
                         'fiber_output_img':analysis.analyze_image(speckle)}
                
                # update and save the corresponding metadata of the image
                meta = {
                        "image_id":str(time.time_ns()), 
                        "capture_time":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "purpose":experiment.get("purpose", None), 
                        "images_per_sample":experiment.get("images_per_sample", None),  
                        "is_params":experiment.get("is_params", False), 
                        "is_calibration":experiment.get("is_calibration", False), 
                        "img_stats":stats,
                        "image_descriptions":json.dumps({**({"simulation_img": img_size} if experiment.get("include_simulation", False) else {}),
                                                         "ground_truth_img": img_size, "fiber_output_img": img_size}),
                        "image_path":relative_path,  
                        "config_id":config_id,
                        "batch":batch,
                        "comments":comment
                        }
                ImageMeta.set_metadata(meta)
                cv2.imwrite(image_path, image)
                DB.sql_execute(ImageMeta.to_sql_insert("mmf_dataset_metadata")) 
            DMD.free_memory()
            
except Exception as e:
    print(f"An error occured, data collection stopped. {e}")
    traceback.print_exc()
    
# update actual number of images captured, excluding the temporal shift check images
sql = """
    SELECT 
        meta.batch AS batch, 
        COUNT(meta.id) - SUM(CASE WHEN meta.comments = 'temporal_shift_check' THEN 1 ELSE 0 END) AS total_images
    FROM 
        mmf_dataset_metadata AS meta
    LEFT JOIN 
        mmf_experiment_config AS conf
    ON 
        meta.config_id = conf.id
    GROUP BY 
        meta.batch
"""
df = DB.sql_select(sql)
sql = DB.batch_update("mmf_experiment_config", "batch", df)
DB.sql_execute(sql, multiple=True)

# ============================
# Close everything properly
# ============================
DB.close()  
DMD.end()
MANAGER.end()