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
import threading
from remoteJapcAccess import *

from dataclasses import dataclass

@dataclass
class CLEARStatus:
    QFD0880: float |None = None
    QDD0870: float |None = None
    DHJ0840: float |None = None
    DVJ0840: float |None = None
    
    @classmethod
    def from_remote_japc(cls, clear_data: dict):
        return cls(
            QFD0880=clear_data.get('CA.QFD0880', None),
            QDD0870=clear_data.get('CA.QDD0870', None),
            DHJ0840=clear_data.get('CA.DHJ0840', None),
            DVJ0840=clear_data.get('CA.DVJ0840', None),
        )
    
    

# ============================
# Dataset Parameters
# ============================
conf = {
    'config_crop_area': False,  # set to True to select crop areas and stop the process
    'camera_order_flip': True,  # camera order flip
    'cam_schedule_time': int(500 * 1e6),  # camera schedule time in milliseconds 500
    'base_resolution': (512, 512),  # base resolution for all images (256, 256)
    'number_of_images': 25000,  # simulation: number of images to generate in this batch
    'number_of_test': 500,  # left none for all images
    'number_of_minst': 100,
    'temporal_shift_freq': 50,  # simulation: temporal shift frequency
    'temporal_shift_intensity': 20,  # simulation: temporal shift check intensity
    'dmd_dim': 1024,  # DMD working square area resolution
    'dmd_rotation': DMD_ROTATION_ANGLE,  # DMD rotation angle for image orientation correction
    'horizontal_flip': True,  # horizontal flip for all images
    'vertical_flip': False,   # vertical flip for all images
    'dmd_bitDepth': 8,  # DMD bit depth
    'dmd_picture_time': 20000,  # DMD picture time in microseconds, corresponds to 50 Hz -> 20000, 10 Hz -> 100000
    'dmd_alp_version': '4.3',  # DMD ALP version
    'crop_areas': [((625, 226), (1299, 900)), ((2273, 7), (3447, 1181))],  # crop areas for the camera images, need to be square
    'sim_pattern_max_num': 100,  # simulation: maximum number of distributions in the simulation
    'sim_fade_rate': 0.96,  # simulation: the probability of a distribution to disappear
    'sim_std_1': 0.02, # simulation: lower indication of std   0.03
    'sim_std_2': 0.25, # simulation: higher indication of std   0.2
    'sim_max_intensity': 100, # simulation: peak pixel intensity in a single distribution
    'sim_dim': 512,   # simulation: simulated image resolution
    
    'dct_dim': (32, 32),  # DCT basis dimension
    'dct_value_range': (0.0, 70.0),  # remap DCT basis values to [0, 255]

    # ----------------------------
    # Camera-only periodic experiment (new)
    # ----------------------------
    'camera_only_enable': False,  # set to True to enable camera-only periodic acquisition experiment
    'camera_only_samples': 50,  # number of images to capture in camera-only experiment
    'camera_only_schedule_time': int(500 * 1e6),  # schedule time for camera-only experiment (same units as cam_schedule_time)
    
    # ----------------------------
    # Real Beamtime Parameters
    # ----------------------------
    'set_magnets': True,  # whether to set magnets before each acquisition
}

# ============================
# External Trigger Controller
# ============================
class TriggerController:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._event = threading.Event()
        if not self.enabled:
            # if trigger not enabled, never block
            self._event.set()

    def wait(self):
        """Block until a trigger arrives, if enabled."""
        if not self.enabled:
            return
        self._event.wait()
        self._event.clear()

    def fire(self):
        """Call this when an external signal happens."""
        self._event.set()

# enable only when you want external control (e.g. camera_only mode)
# TRIGGER = TriggerController(enabled=conf.get("camera_only_enable", False))
TRIGGER = TriggerController(enabled=False)

# An example of control signal come from an server API endpoint

# import urllib.request

# API_URL = "http://your-server/api/trigger"  # <-- your real URL

# def api_trigger_loop():
#     last_ready = False
#     while True:
#         try:
#             with urllib.request.urlopen(API_URL, timeout=2.0) as resp:
#                 data = json.load(resp)
#             ready = bool(data.get("ready", False))

#             # rising edge: False -> True => one trigger
#             if ready and not last_ready:
#                 TRIGGER.fire()

#             last_ready = ready
#         except Exception as e:
#             print("Trigger poll error:", e)

#         time.sleep(1.0)  # check every 1 s

# # start only when external trigger is desired
# if conf.get("camera_only_enable", False):
#     threading.Thread(target=api_trigger_loop, daemon=True).start()



# ============================
# Hardware/Software Initialization
# ============================
# DMD Initialization (real hardware if available, otherwise dummy)
try:
    DMD = dmd.ViALUXDMD(ALP4(version=conf['dmd_alp_version']))
    DMD.set_pictureTime(conf['dmd_picture_time'])
except Exception as e:
    print(f"Warning: DMD initialization failed ({e}). Using DummyDMD (camera-only mode).")

    class DummyDMD:
        def set_pictureTime(self, *args, **kwargs):
            pass

        def display_image(self, *args, **kwargs):
            pass

        def get_metadata(self):
            return {"type": "DummyDMD", "status": "not_initialized"}

        def free_memory(self):
            pass

        def end(self):
            pass

    DMD = DummyDMD()

# generate_upward_arrow(), dmd_calibration_pattern_generation()   generate_circle_fiber_coupling_pattern(line_width=20)
# calibration_img = np.ones((256, 256)) * 100
# calibration_img = simulation.generate_radial_gradient()
# calibration_img = simulation.generate_upward_arrow()
# calibration_img = simulation.generate_up_left_arrow()
calibration_img = simulation.dmd_calibration_pattern_generation()
calibration_img = simulation.macro_pixel(calibration_img, size=int(conf['dmd_dim']/calibration_img.shape[0])) 
DMD.display_image(dmd.dmd_img_adjustment(calibration_img, conf['dmd_dim'], angle=conf['dmd_rotation'], horizontal_flip=conf['horizontal_flip'], vertical_flip=conf['vertical_flip']))

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
    DMD.display_image(dmd.dmd_img_adjustment(calibration_img, conf['dmd_dim'], angle=conf['dmd_rotation'], horizontal_flip=conf['horizontal_flip'], vertical_flip=conf['vertical_flip']))
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
    paths = utils.select_random_elements(paths, conf['number_of_test'])
process_funcs = [utils.rgb_to_grayscale, utils.split_image, lambda x : x[0].astype(np.uint8)]

if conf['number_of_minst']:
    imgs_array = utils.select_random_elements(read_MNIST_images(minst_path), conf['number_of_minst'])
    minst_len = len(imgs_array)
    imgs_array = utils.list_to_generator(imgs_array)
    print(type(imgs_array))


# create a queue of image sources
# simulation_config, other_notes, experiment_description, image_source, purpose, images_per_sample, is_params, is_calibration
queue = []
queue.append({'experiment_description':'empty (only black) image', 
              'purpose':'calibration',
              'image_source':'simulation',
              'images_per_sample':2,
              'is_calibration':True,
              'data':[np.ones((256, 256)) * 0],
              'len':1})

queue.append({'experiment_description':'calibration image', 
              'purpose':'calibration',
              'image_source':'simulation',
              'images_per_sample':2,
              'is_calibration':True,
              'data':[simulation.dmd_calibration_pattern_generation()],
              'len':1}) 
# queue.append({'experiment_description': 'full screen image',
#               'purpose':'intensity_full',
#               'image_source':'simulation',
#               'images_per_sample':2,
#               'data': [np.ones(conf['base_resolution']) * 100],
#               'len':1}) 

queue.append({'experiment_description':'position based coupling intensity',
              'purpose':'intensity_position',
              'image_source':'simulation',
              'images_per_sample':2,
              'data':simulation.moving_blocks_generator(size=conf['base_resolution'][0], block_size=64, intensity=255),
              'len':64}) 
# queue.append({'experiment_description':'position based coupling intensity',
#               'purpose':'intensity_position',
#               'image_source':'simulation',
#               'images_per_sample':2,
#               'data':simulation.moving_blocks_generator(size=conf['base_resolution'][0], block_size=32, intensity=255),
#               'len':256}) 
# queue.append({'experiment_description':'position based coupling intensity',
#               'purpose':'intensity_position',
#               'image_source':'simulation',
#               'images_per_sample':2,
#               'data':simulation.moving_blocks_generator(size=conf['base_resolution'][0], block_size=16, intensity=255),
#               'len':1024}) 

# queue.append({'experiment_description':'MINST for fun',
#               'purpose':'fun',
#               'images_per_sample':2,
#               'image_source':'MINST',
#               'data':simulation.temporal_shift(conf['temporal_shift_freq'], conf['temporal_shift_intensity'])(utils.identity)(imgs_array),
#               'len':minst_len + utils.ceil_int_div(minst_len, conf['temporal_shift_freq'])}) 

queue.append({'experiment_description':'DCT basis patterns',
              'purpose':'Orthogonal_basis',
              'image_source':'simulation',
              'images_per_sample':2,
              'data':basis.make_dct(shape = conf['dct_dim'], value_range=conf['dct_value_range']).generator(),
              'len':conf['dct_dim'][0] * conf['dct_dim'][1]}) 

queue.append({'experiment_description':'local real beam image for evaluation',
              'purpose':'testing',
              'images_per_sample':2,
              'image_source':'e-beam',
              'is_params':True,
              'data':simulation.temporal_shift(conf['temporal_shift_freq'], conf['temporal_shift_intensity'])(simulation.read_local_generator)(paths, process_funcs),
              'len':len(paths) + utils.ceil_int_div(len(paths), conf['temporal_shift_freq'])}) 

queue.append({'experiment_description':'2d multi-gaussian distributions simulation',
              'purpose':'training',
              'image_source':'simulation',
              'images_per_sample':2,
              'simulation_config':CANVAS.get_metadata(),
              'other_notes':{key: value for key, value in conf.items() if 'sim' in key},
              'data':simulation.temporal_shift(conf['temporal_shift_freq'], conf['temporal_shift_intensity'])(simulation.canvas_generator)(CANVAS, conf),
              'len':conf['number_of_images'] + utils.ceil_int_div(conf['number_of_images'], conf['temporal_shift_freq'])}) 



# ============================
# Camera-only periodic experiment queue (No DMD needed)
# ============================
if conf.get('camera_only_enable', False):
    queue = []

    def camera_only_generator(num_samples, base_resolution):
        """
        Simple generator that yields blank images with the given base_resolution.
        These are only used as placeholders to keep the framework identical;
        the real information comes from the cameras via scheduled action commands.
        """
        for _ in range(num_samples):
            yield np.zeros(base_resolution, dtype=np.uint8)

    queue.append({
        'experiment_description': 'camera-only periodic acquisition on CLEAR to check if the aquisition scripts work properly',
        'purpose': 'first test at CLEAR',
        'image_source': 'CLEAR e-beam',  
        'image_device': 'Chromox scintillator', 
        'images_per_sample': 2,  # still two cameras
        'data': camera_only_generator(conf['camera_only_samples'], conf['base_resolution']),
        'len': conf['camera_only_samples'],
        # per-experiment schedule time (period), same units as conf['cam_schedule_time']
        'cam_schedule_time': conf.get('camera_only_schedule_time', conf['cam_schedule_time']),
        # no simulation added into the saved image
        'include_simulation': False,
        'is_calibration': False,
        'is_params': True,
    })
    
    # queue.append({
    #     'experiment_description': 'CLEAR random beam with parameters: ...',
    #     'purpose': 'testset',
    #     'image_source': 'CLEAR e-beam',  
    #     'image_device': 'Chromox scintillator', 
    #     'images_per_sample': 2,  # still two cameras
    #     'data': camera_only_generator(conf['camera_only_samples'], conf['base_resolution']),
    #     'len': conf['camera_only_samples'],
    #     # per-experiment schedule time (period), same units as conf['cam_schedule_time']
    #     'cam_schedule_time': conf.get('camera_only_schedule_time', conf['cam_schedule_time']),
    #     # no simulation added into the saved image
    #     'include_simulation': False,
    #     'is_calibration': False,
    #     'is_params': True,
    # })


# ============================
# data collection pipeline
# ============================
try:
    for experiment in queue:
        # Setting up experiment metadata
        batch = (DB.get_max("mmf_dataset_metadata", "batch") or 0) + 1  # get the current batch number
        experiment_metadata = {
            "experiment_location": "Optical Lab, CERN, Geneva, Switzerland",
            "experiment_date": datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
            "image_device": experiment.get("image_device", "dmd"),  # scintillation-screen, dmd, slm, led, OTR.
            "fiber_config": {
                "fiber_length": "15 m",
                "fiber_name": "1500 micrometer Core-diameter Step-Index Multimode Fiber Patch Cable",
                "fiber_url": "FP1500ERT",  # FP1500ERT  FT600UMT
            },
            "camera_config": MANAGER.get_metadata(), # assume not changing during entire experiment
            "other_config": {
                "dmd_config": DMD.get_metadata(), 
                "simulation_config": experiment.get("simulation_config", None),
                "light_source": "Broadband LED with 694nm +- 10nm bandpass filter",  # e-beam, proton-beam, LED, laser
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
            # Append the new batch number to the existing batch list
            print("Notice: The experiment metadata already exists in the database. Appending batch number.")
            existing_batch_sql = f"SELECT batch FROM mmf_experiment_config WHERE hash = '{ConfMeta.get_hash()}'"
            existing_batch_df = DB.sql_select(existing_batch_sql)
            existing_batch = existing_batch_df.iloc[0]['batch'] if not existing_batch_df.empty else ""
            
            # Append new batch to existing batch list
            if existing_batch:
                new_batch_list = f"{existing_batch}, {batch}"
            else:
                new_batch_list = str(batch)
            
            # Update the batch field with the appended list
            update_batch_sql = f"UPDATE mmf_experiment_config SET batch = '{new_batch_list}' WHERE hash = '{ConfMeta.get_hash()}'"
            DB.sql_execute(update_batch_sql)
        
        # Get the config_id for the current experiment configuration by hash
        config_id_sql = f"SELECT id FROM mmf_experiment_config WHERE hash = '{ConfMeta.get_hash()}'"
        config_id_df = DB.sql_select(config_id_sql)
        config_id = config_id_df.iloc[0]['id'] if not config_id_df.empty else None

        # print to indicate the current experiment name
        print(f"---> Starting experiment: {experiment['experiment_description']}")
        for img in tqdm(experiment['data'], total=experiment['len']):
            comment = None
            if isinstance(img, tuple): # check if the generator returns a tuple (image, comment)
                img, comment = img

            # Only use img for DMD display if it is not None
            if img is not None:
                display = img.copy()
                display = simulation.macro_pixel(display, size=int(conf['dmd_dim']/display.shape[0])) 
                display = dmd.dmd_img_adjustment(display, conf['dmd_dim'], angle=conf['dmd_rotation'], horizontal_flip=conf['horizontal_flip'], vertical_flip=conf['vertical_flip'])
                DMD.display_image(display)  # if loading too fast, the DMD might report memory error
            
            # capture the image from the cameras (Scheduled action command)
            schedule_time = experiment.get("cam_schedule_time", conf['cam_schedule_time'])
            
            # external trigger gate (only blocks if camera_only_enable=True)
            TRIGGER.wait()
            
            # ------------------------------------- logic for real beam ----------------------------------------
            # set magnets
            if conf['set_magnets']:
                # example: set random magnet currents for CLEAR
                setMagnetsCurrent({
                    'CA.QFD0880': np.random.uniform(1.0, 20.0),
                    'CA.QDD0870': np.random.uniform(1.0, 20.0),
                    'CA.DHJ0840': np.random.uniform(-6.0, 6.0),
                    'CA.DVJ0840': np.random.uniform(-6.0, 6.0),
                })
            
                # wait until all magnets are not busy
                magnet_names = ['CA.QFD0880', 'CA.QDD0870', 'CA.DHJ0840', 'CA.DVJ0840']
                while True:
                    statuses = getMagnetsStatus(magnet_names)
                    if not any(statuses.values()):  # break if all magnets are not busy
                        break
                    time.sleep(0.5)
                        
                # acquire status
                try:
                    clear_status = CLEARStatus.from_remote_japc(getMagnetsCurrent(['CA.QFD0880', 'CA.QDD0870', 'CA.DHJ0840', 'CA.DVJ0840']))
                except:
                    clear_status = CLEARStatus()
                    
                # save status as part of metadata
                beam_settings = {
                    "CLEAR_magnets": clear_status.__dict__,
                }
            # --------------------------------------------------------------------------------------------------------
            
            image = MANAGER.schedule_action_command(schedule_time) 
            if image is not None:
                img_size = (image.shape[0], int(image.shape[1]//2))  
                if experiment.get("include_simulation", False) and img is not None: # optional, add the very original image
                    original_image = cv2.resize(img, (image.shape[0],image.shape[0])) 
                    image = np.hstack((original_image, image))
                filename = str(time.time_ns()) + '.png'
                image_path = save_dir + '/' + filename # absolute path save on the local machine
                relative_path = '/'.join(['dataset', str(batch), filename]) # relative path save in the database
                # crop the image to regions of interest
                image = processing.crop_image_from_coordinates(image, conf['crop_areas'])
                # image statistics info
                ground_truth, fiber_output = utils.split_image(image)
                ground_truth_stats = analysis.analyze_image(ground_truth)
                fiber_output_stats = analysis.analyze_image(fiber_output)
                
                # calculate saturation status
                is_saturated_ground_truth = 1 if ground_truth_stats.get("max_intensity", 0) >= 255 else 0
                is_saturated_fiber_output = 1 if fiber_output_stats.get("max_intensity", 0) >= 255 else 0
                
                # calculate coupling efficiency
                coupling_efficiency = None
                if ground_truth_stats.get("total_intensity", 0) > 0:
                    coupling_efficiency = fiber_output_stats.get("total_intensity", 0) / ground_truth_stats.get("total_intensity", 1)
                
                # update and save the corresponding metadata of the image
                meta = {
                        "image_id":str(time.time_ns()), 
                        "capture_time":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "purpose":experiment.get("purpose", None), 
                        "images_per_sample":experiment.get("images_per_sample", None),  
                        "is_params":experiment.get("is_params", False), 
                        "is_calibration":experiment.get("is_calibration", False), 
                        "is_saturated_ground_truth":is_saturated_ground_truth,
                        "is_saturated_fiber_output":is_saturated_fiber_output,
                        "coupling_efficiency":coupling_efficiency,
                        "ground_truth_img_stat":ground_truth_stats,
                        "fiber_output_img_stats":fiber_output_stats,
                        "image_descriptions":json.dumps({**({"simulation_img": img_size} if experiment.get("include_simulation", False) else {}),
                                                         "ground_truth_img": img_size, "fiber_output_img": img_size}),
                        "image_path":relative_path,  
                        "config_id":config_id,
                        "batch":batch,
                        "comments":comment,
                        "beam_settings": beam_settings,  # beam settings at the time of image capture
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
        conf.id AS id,
        COUNT(meta.id) - SUM(CASE WHEN meta.comments = 'temporal_shift_check' THEN 1 ELSE 0 END) AS total_images
    FROM 
        mmf_dataset_metadata AS meta
    LEFT JOIN 
        mmf_experiment_config AS conf
    ON 
        meta.config_id = conf.id
    GROUP BY 
        meta.config_id
"""
df = DB.sql_select(sql)
sql = DB.batch_update("mmf_experiment_config", "id", df)
DB.sql_execute(sql, multiple=True)

# ============================
# Close everything properly
# ============================
DB.close()  
DMD.end()
MANAGER.end()
