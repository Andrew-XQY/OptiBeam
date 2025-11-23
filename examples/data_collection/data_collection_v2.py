"""
    Enhanced data collection script with improved UI and dual camera support
    - Fixed 1 Hz update rate (configurable)
    - Tkinter input panel for parameters (no CV2 sliders)
    - Same metadata display as filter_test.py
    - Core logic preserved from data_collection_dmd.py
"""
from conf import *
from ALP4 import *
from tqdm import tqdm
import datetime, time
import cv2, json
import subprocess
import traceback
import threading
import tkinter as tk
from tkinter import messagebox
from remoteJapcAccess import *
from collections import deque

from dataclasses import dataclass


class MaxPixelBuffer:
    """Buffer to track maximum pixel value over a time window"""
    
    def __init__(self, window_seconds=10.0):
        """
        Initialize the buffer
        
        Args:
            window_seconds: float - time window in seconds (default 10.0)
        """
        self.window_seconds = window_seconds
        self.buffer = deque()  # Store (timestamp, max_pixel_value) tuples
    
    def add_value(self, max_pixel_value):
        """
        Add a new max pixel value to the buffer
        
        Args:
            max_pixel_value: float - maximum pixel value from current frame
        """
        current_time = time.time()
        self.buffer.append((current_time, max_pixel_value))
        
        # Remove old values outside the time window
        cutoff_time = current_time - self.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff_time:
            self.buffer.popleft()
    
    def get_max(self):
        """
        Get the maximum pixel value within the time window
        
        Returns:
            float - maximum pixel value in the buffer, or None if buffer is empty
        """
        if not self.buffer:
            return None
        return max(value for _, value in self.buffer)
    
    def get_buffer_duration(self):
        """
        Get the actual duration of data in the buffer
        
        Returns:
            float - duration in seconds, or 0 if buffer is empty
        """
        if len(self.buffer) < 2:
            return 0.0
        return self.buffer[-1][0] - self.buffer[0][0]


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
# Configuration Parameters
# ============================
conf = {
    'config_crop_area': False,
    'camera_order_flip': True,
    'cam_schedule_time': int(500 * 1e6),  # camera schedule time in microseconds
    'base_resolution': (512, 512),
    'number_of_images': 25000,
    'number_of_test': 500,
    'number_of_minst': 100,
    'temporal_shift_freq': 50,
    'temporal_shift_intensity': 20,
    'dmd_dim': 1024,
    'dmd_rotation': DMD_ROTATION_ANGLE,
    'horizontal_flip': True,
    'vertical_flip': False,
    'dmd_bitDepth': 8,
    'dmd_picture_time': 20000,
    'dmd_alp_version': '4.3',
    'crop_areas': [((625, 226), (1299, 900)), ((2273, 7), (3447, 1181))],
    'sim_pattern_max_num': 100,
    'sim_fade_rate': 0.96,
    'sim_std_1': 0.02,
    'sim_std_2': 0.25,
    'sim_max_intensity': 100,
    'sim_dim': 512,
    'dct_dim': (32, 32),
    'dct_value_range': (0.0, 70.0),
    
    
    'camera_only_enable': False,
    'camera_only_samples': 50,
    'camera_only_schedule_time': int(500 * 1e6),
    'set_magnets': False,  # Set to True when doing real beamtime experiments
    
    # UI and Display Parameters
    'preview_update_rate_hz': 1.0,  # Preview window update rate (Hz)
    'preview_scale_factor': 0.5,    # Display scale factor
    'text_scale': 0.8,              # Text overlay scale
    'max_buffer_seconds': 10.0,     # Time window for max pixel tracking (seconds)
    'camera_window_scale': 1.2,     # CV2 camera setup window scale ratio (e.g., 0.4, 0.7, 1.0, 1.5)
}


# ============================
# Helper Functions (from filter_test.py)
# ============================
def analyze_frame_properties(image, normalize_range=(0, 100)):
    """Analyze frame properties and normalize them to a specified range"""
    max_pixel = np.max(image)
    total_sum = np.sum(image, dtype=np.float64)
    
    min_val, max_val = normalize_range
    max_possible = 255 if image.dtype == np.uint8 else 65535
    normalized_max = min_val + (max_pixel / max_possible) * (max_val - min_val)
    
    max_possible_sum = image.size * max_possible
    normalized_sum = min_val + (total_sum / max_possible_sum) * (max_val - min_val)
    
    properties = {
        'Max Pixel Value': f'{normalized_max:.2f}',
        'Total Sum': f'{normalized_sum:.2f}'
    }
    
    return properties


def get_camera_parameters_display(manager):
    """Get camera parameters for display overlay"""
    cam_meta = manager.get_metadata()
    exposure_us = cam_meta.get('exposure_time', 0)
    exposure_ms = exposure_us / 1000.0
    
    return {
        'Exposure (us)': f'{exposure_us}',
        'Exposure (ms)': f'{exposure_ms:.2f}',
        'Gain': f'{cam_meta.get("gain", 0)}',
        'Gamma': f'{cam_meta.get("gamma", 1.0):.2f}',
        'Num Cameras': f'{len(manager.cameras)}',
    }


def add_text_to_image(image, camera_params, frame_properties, text_scale=1.0):
    """Add camera parameters and frame properties as text overlay"""
    if len(image.shape) == 2:
        img_bgr = np.stack([image, image, image], axis=2)
    else:
        img_bgr = image.copy()
    
    y_position = int(30 * text_scale)
    line_spacing = int(35 * text_scale)
    thickness = max(1, int(2 * text_scale))
    
    # Frame properties (top left, white)
    for key, value in frame_properties.items():
        text = f'{key}: {value}'
        cv2.putText(img_bgr, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                   text_scale, (0, 0, 0), thickness + 2)
        cv2.putText(img_bgr, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                   text_scale, (255, 255, 255), thickness)
        y_position += line_spacing
    
    # Camera parameters (bottom right, blue)
    y_position = img_bgr.shape[0] - int(20 * text_scale)
    for key, value in reversed(list(camera_params.items())):
        text = f'{key}: {value}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)[0]
        x_position = img_bgr.shape[1] - text_size[0] - 10
        cv2.putText(img_bgr, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                   text_scale, (0, 0, 0), thickness + 2)
        cv2.putText(img_bgr, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                   text_scale, (255, 0, 0), thickness)
        y_position -= line_spacing
    
    return img_bgr


def image_resize(img, scale_percent):
    """Resize image by a scaling factor"""
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    return cv2.resize(img, (width, height))


# ============================
# Parameter Input Panel (Tkinter)
# ============================
class ParameterInputPanel:
    def __init__(self, conf):
        self.conf = conf
        self.window = tk.Tk()
        self.window.title("Data Collection Parameters")
        self.window.geometry("400x350")
        
        # Camera Schedule Time
        tk.Label(self.window, text="Camera Schedule Time (microseconds):", font=("Arial", 10)).pack(pady=5)
        self.schedule_entry = tk.Entry(self.window, font=("Arial", 11), width=20)
        self.schedule_entry.insert(0, str(conf['cam_schedule_time']))
        self.schedule_entry.pack(pady=2)
        
        # DMD Picture Time
        tk.Label(self.window, text="DMD Picture Time (microseconds):", font=("Arial", 10)).pack(pady=5)
        self.dmd_time_entry = tk.Entry(self.window, font=("Arial", 11), width=20)
        self.dmd_time_entry.insert(0, str(conf['dmd_picture_time']))
        self.dmd_time_entry.pack(pady=2)
        
        # Preview Update Rate
        tk.Label(self.window, text="Preview Update Rate (Hz):", font=("Arial", 10)).pack(pady=5)
        self.update_rate_entry = tk.Entry(self.window, font=("Arial", 11), width=20)
        self.update_rate_entry.insert(0, str(conf['preview_update_rate_hz']))
        self.update_rate_entry.pack(pady=2)
        
        # Preview Scale Factor
        tk.Label(self.window, text="Preview Scale Factor (0.1-1.0):", font=("Arial", 10)).pack(pady=5)
        self.scale_entry = tk.Entry(self.window, font=("Arial", 11), width=20)
        self.scale_entry.insert(0, str(conf['preview_scale_factor']))
        self.scale_entry.pack(pady=2)
        
        # Apply Button
        self.apply_button = tk.Button(self.window, text="Apply Settings", 
                                      command=self.apply_settings,
                                      font=("Arial", 11), bg="lightgreen", width=20)
        self.apply_button.pack(pady=15)
        
        # Status Label
        self.status_label = tk.Label(self.window, text="Ready", 
                                     font=("Arial", 9), fg="blue")
        self.status_label.pack(pady=5)
        
        self.window.attributes('-topmost', True)
        
    def apply_settings(self):
        try:
            # Validate and apply settings
            schedule_time = int(self.schedule_entry.get())
            dmd_time = int(self.dmd_time_entry.get())
            update_rate = float(self.update_rate_entry.get())
            scale_factor = float(self.scale_entry.get())
            
            if schedule_time <= 0 or dmd_time <= 0:
                raise ValueError("Times must be positive!")
            if update_rate <= 0 or update_rate > 100:
                raise ValueError("Update rate must be between 0 and 100 Hz!")
            if scale_factor <= 0 or scale_factor > 1.0:
                raise ValueError("Scale factor must be between 0.1 and 1.0!")
            
            self.conf['cam_schedule_time'] = schedule_time
            self.conf['dmd_picture_time'] = dmd_time
            self.conf['preview_update_rate_hz'] = update_rate
            self.conf['preview_scale_factor'] = scale_factor
            
            self.status_label.config(text="Settings Applied Successfully!", fg="green")
            print(f"Settings updated: schedule={schedule_time}us, dmd_time={dmd_time}us, rate={update_rate}Hz, scale={scale_factor}")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            self.status_label.config(text="Error in input!", fg="red")
    
    def update(self):
        """Update the Tkinter window"""
        try:
            self.window.update()
            return True
        except tk.TclError:
            return False


# ============================
# Preview Window Manager
# ============================
class PreviewWindowManager:
    def __init__(self, conf, manager):
        self.conf = conf
        self.manager = manager
        self.window_name_cam1 = 'Camera 1 (Ground Truth)'
        self.window_name_cam2 = 'Camera 2 (Fiber Output)'
        cv2.namedWindow(self.window_name_cam1)
        cv2.namedWindow(self.window_name_cam2)
        self.last_update_time = time.time()
        
        # Initialize max pixel buffers for each camera
        self.max_buffer_cam1 = MaxPixelBuffer(window_seconds=conf['max_buffer_seconds'])
        self.max_buffer_cam2 = MaxPixelBuffer(window_seconds=conf['max_buffer_seconds'])
        
    def should_update(self):
        """Check if enough time has passed for the next update"""
        current_time = time.time()
        interval = 1.0 / self.conf['preview_update_rate_hz']
        if current_time - self.last_update_time >= interval:
            self.last_update_time = current_time
            return True
        return False
    
    def display_image(self, image):
        """Display individual camera images with metadata overlay"""
        if image is None:
            return
        
        # Split the combined image into two cameras
        # Assuming horizontal concatenation: [cam1 | cam2]
        height, width = image.shape[:2]
        mid_point = width // 2
        cam1_img = image[:, :mid_point]
        cam2_img = image[:, mid_point:]
        
        # Get camera parameters (shared across cameras)
        camera_params = get_camera_parameters_display(self.manager)
        
        # Process Camera 1 (Ground Truth)
        frame_properties_cam1 = analyze_frame_properties(cam1_img)
        current_max_cam1 = np.max(cam1_img)
        self.max_buffer_cam1.add_value(current_max_cam1)
        windowed_max_cam1 = self.max_buffer_cam1.get_max()
        if windowed_max_cam1 is not None:
            buffer_duration_cam1 = self.max_buffer_cam1.get_buffer_duration()
            windowed_max_normalized_cam1 = (windowed_max_cam1 / 255.0) * 100.0
            frame_properties_cam1[f'{self.conf["max_buffer_seconds"]}s Max Pixel'] = \
                f"{windowed_max_normalized_cam1:.1f} ({buffer_duration_cam1:.1f}s)"
        
        # Add text overlay for Camera 1
        cam1_with_info = add_text_to_image(cam1_img, camera_params, frame_properties_cam1, 
                                           self.conf['text_scale'])
        
        # Process Camera 2 (Fiber Output)
        frame_properties_cam2 = analyze_frame_properties(cam2_img)
        current_max_cam2 = np.max(cam2_img)
        self.max_buffer_cam2.add_value(current_max_cam2)
        windowed_max_cam2 = self.max_buffer_cam2.get_max()
        if windowed_max_cam2 is not None:
            buffer_duration_cam2 = self.max_buffer_cam2.get_buffer_duration()
            windowed_max_normalized_cam2 = (windowed_max_cam2 / 255.0) * 100.0
            frame_properties_cam2[f'{self.conf["max_buffer_seconds"]}s Max Pixel'] = \
                f"{windowed_max_normalized_cam2:.1f} ({buffer_duration_cam2:.1f}s)"
        
        # Add text overlay for Camera 2
        cam2_with_info = add_text_to_image(cam2_img, camera_params, frame_properties_cam2, 
                                           self.conf['text_scale'])
        
        # Resize for display
        cam1_display = image_resize(cam1_with_info, self.conf['preview_scale_factor'])
        cam2_display = image_resize(cam2_with_info, self.conf['preview_scale_factor'])
        
        # Display each camera in separate windows
        cv2.imshow(self.window_name_cam1, cam1_display)
        cv2.imshow(self.window_name_cam2, cam2_display)
        cv2.waitKey(1)  # Non-blocking
    
    def close(self):
        cv2.destroyAllWindows()


# ============================
# Hardware Initialization
# ============================
# DMD Initialization
try:
    DMD = dmd.ViALUXDMD(ALP4(version=conf['dmd_alp_version']))
    DMD.set_pictureTime(conf['dmd_picture_time'])
    print("DMD initialized successfully.")
except Exception as e:
    print(f"Warning: DMD initialization failed ({e}). Using DummyDMD.")
    
    class DummyDMD:
        def set_pictureTime(self, *args, **kwargs): pass
        def display_image(self, *args, **kwargs): pass
        def get_metadata(self): return {"type": "DummyDMD", "status": "not_initialized"}
        def free_memory(self): pass
        def end(self): pass
    
    DMD = DummyDMD()

# Display calibration pattern
calibration_img = simulation.dmd_calibration_pattern_generation()
calibration_img = simulation.macro_pixel(calibration_img, size=int(conf['dmd_dim']/calibration_img.shape[0])) 
DMD.display_image(dmd.dmd_img_adjustment(calibration_img, conf['dmd_dim'], 
                                         angle=conf['dmd_rotation'], 
                                         horizontal_flip=conf['horizontal_flip'], 
                                         vertical_flip=conf['vertical_flip']))

# Cameras Initialization
MANAGER = camerav2.MultiBaslerCameraManager()
# Set initial flip preference (can be changed in config window)
MANAGER.flip = conf['camera_order_flip']
# Initialize with configuration window - pass UI parameters
config_params = {
    'update_rate_hz': conf['preview_update_rate_hz'],
    'scale_factor': conf['preview_scale_factor'],
    'text_scale': conf['text_scale'],
    'save_dir': 'C:\\Users\\qiyuanxu\\Desktop\\',  # Save directory for 's' key during setup
    'window_scale': conf['camera_window_scale'],  # CV2 window scale ratio
}
MANAGER.initialize(show_config_window=True, config_params=config_params)
MANAGER.synchronization()
print(f"Initialized {len(MANAGER.cameras)} cameras.")


# ============================
# Crop Area Selection (Optional)
# ============================
if conf['config_crop_area']:
    calibration_img = simulation.dmd_calibration_pattern_generation()
    calibration_img = simulation.macro_pixel(calibration_img, size=int(conf['dmd_dim']/calibration_img.shape[0])) 
    DMD.display_image(dmd.dmd_img_adjustment(calibration_img, conf['dmd_dim'], 
                                             angle=conf['dmd_rotation'], 
                                             horizontal_flip=conf['horizontal_flip'], 
                                             vertical_flip=conf['vertical_flip']))
    test_img = MANAGER.schedule_action_command(conf['cam_schedule_time'])
    test_img = processing.add_grid(test_img, partitions=50)
    crop_areas = processing.select_crop_areas_corner(test_img, num=2, scale_factor=0.4) 
    sys.exit(f"Crop areas selected: {crop_areas} \nProcedure completed.")


# ============================
# Queue Setup (Same as original)
# ============================
DB = database.SQLiteDB(DATABASE_ROOT)
ImageMeta = metadata.ImageMetadata()
ConfMeta = metadata.ConfigMetaData()

CANVAS = simulation.DynamicPatterns(conf['sim_dim'], conf['sim_dim'])
CANVAS._distributions = [simulation.StaticGaussianDistribution(CANVAS) for _ in range(conf['sim_pattern_max_num'])] 

paths = utils.get_all_file_paths(path_to_images) 
if conf['number_of_test']:
    paths = utils.select_random_elements(paths, conf['number_of_test'])
process_funcs = [utils.rgb_to_grayscale, utils.split_image, lambda x : x[0].astype(np.uint8)]

if conf['number_of_minst']:
    imgs_array = utils.select_random_elements(read_MNIST_images(minst_path), conf['number_of_minst'])
    minst_len = len(imgs_array)
    imgs_array = utils.list_to_generator(imgs_array)

# Build queue (same structure as original)
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

queue.append({'experiment_description':'position based coupling intensity',
              'purpose':'intensity_position',
              'image_source':'simulation',
              'images_per_sample':2,
              'data':simulation.moving_blocks_generator(size=conf['base_resolution'][0], block_size=64, intensity=255),
              'len':64}) 

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
# Data Collection Pipeline with UI
# ============================
# Create UI components
param_panel = ParameterInputPanel(conf)
preview_manager = PreviewWindowManager(conf, MANAGER)

print("\n" + "="*50)
print("Data Collection Started")
print(f"Preview Update Rate: {conf['preview_update_rate_hz']} Hz")
print(f"Press ESC in preview window to stop")
print("="*50 + "\n")

try:
    for experiment in queue:
        # Setup experiment metadata (same as original)
        batch = (DB.get_max("mmf_dataset_metadata", "batch") or 0) + 1
        experiment_metadata = {
            "experiment_location": "Optical Lab, CERN, Geneva, Switzerland",
            "experiment_date": datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
            "image_device": experiment.get("image_device", "dmd"),
            "fiber_config": {
                "fiber_length": "15 m",
                "fiber_name": "1500 micrometer Core-diameter Step-Index Multimode Fiber Patch Cable",
                "fiber_url": "FP1500ERT",
            },
            "camera_config": MANAGER.get_metadata(),
            "other_config": {
                "dmd_config": DMD.get_metadata(), 
                "simulation_config": experiment.get("simulation_config", None),
                "light_source": "Broadband LED with 694nm +- 10nm bandpass filter",
                'other_notes': experiment.get("other_notes", None)},
            "purtubations": experiment.get("purtubations", None),
            "radiation" : experiment.get("radiation", None),
            "batch": batch,
            "experiment_description": experiment.get("experiment_description", None), 
            "image_source": experiment.get("image_source", None)
        }

        save_dir = DATASET_ROOT + str(batch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' was created.")
            
        ConfMeta.set_metadata(experiment_metadata) 
        if not DB.record_exists("mmf_experiment_config", "hash", ConfMeta.get_hash()):
            DB.sql_execute(ConfMeta.to_sql_insert("mmf_experiment_config"))
        else:
            print("Notice: The experiment metadata already exists. Appending batch number.")
            existing_batch_sql = f"SELECT batch FROM mmf_experiment_config WHERE hash = '{ConfMeta.get_hash()}'"
            existing_batch_df = DB.sql_select(existing_batch_sql)
            existing_batch = existing_batch_df.iloc[0]['batch'] if not existing_batch_df.empty else ""
            new_batch_list = f"{existing_batch}, {batch}" if existing_batch else str(batch)
            update_batch_sql = f"UPDATE mmf_experiment_config SET batch = '{new_batch_list}' WHERE hash = '{ConfMeta.get_hash()}'"
            DB.sql_execute(update_batch_sql)
        
        config_id_sql = f"SELECT id FROM mmf_experiment_config WHERE hash = '{ConfMeta.get_hash()}'"
        config_id_df = DB.sql_select(config_id_sql)
        config_id = config_id_df.iloc[0]['id'] if not config_id_df.empty else None

        print(f"---> Starting experiment: {experiment['experiment_description']}")
        
        for img in tqdm(experiment['data'], total=experiment['len']):
            # Update UI
            if not param_panel.update():
                print("Parameter panel closed. Stopping...")
                raise KeyboardInterrupt
            
            # Check for ESC key
            key = cv2.waitKey(1)
            if key == 27:
                print("ESC pressed. Stopping...")
                raise KeyboardInterrupt
            
            comment = None
            if isinstance(img, tuple):
                img, comment = img

            # Display on DMD
            if img is not None:
                display = img.copy()
                display = simulation.macro_pixel(display, size=int(conf['dmd_dim']/display.shape[0])) 
                display = dmd.dmd_img_adjustment(display, conf['dmd_dim'], 
                                                angle=conf['dmd_rotation'], 
                                                horizontal_flip=conf['horizontal_flip'], 
                                                vertical_flip=conf['vertical_flip'])
                DMD.display_image(display)
            
            # Capture from cameras
            schedule_time = experiment.get("cam_schedule_time", conf['cam_schedule_time'])
            
            # Set magnets if needed (beamtime)
            beam_settings = {}
            if conf['set_magnets']:
                setMagnetsCurrent({
                    'CA.QFD0880': np.random.uniform(1.0, 20.0),
                    'CA.QDD0870': np.random.uniform(1.0, 20.0),
                    'CA.DHJ0840': np.random.uniform(-6.0, 6.0),
                    'CA.DVJ0840': np.random.uniform(-6.0, 6.0),
                })
                
                magnet_names = ['CA.QFD0880', 'CA.QDD0870', 'CA.DHJ0840', 'CA.DVJ0840']
                while True:
                    statuses = getMagnetsStatus(magnet_names)
                    if not any(statuses.values()):
                        break
                    time.sleep(0.5)
                
                try:
                    clear_status = CLEARStatus.from_remote_japc(getMagnetsCurrent(magnet_names))
                except:
                    clear_status = CLEARStatus()
                
                beam_settings = {"CLEAR_magnets": clear_status.__dict__}
            
            image = MANAGER.schedule_action_command(schedule_time)
            
            # Update preview at fixed rate
            if preview_manager.should_update():
                preview_manager.display_image(image)
            
            if image is not None:
                img_size = (image.shape[0], int(image.shape[1]//2))
                if experiment.get("include_simulation", False) and img is not None:
                    original_image = cv2.resize(img, (image.shape[0], image.shape[0])) 
                    image = np.hstack((original_image, image))
                
                filename = str(time.time_ns()) + '.png'
                image_path = save_dir + '/' + filename
                relative_path = '/'.join(['dataset', str(batch), filename])
                
                # Crop and analyze
                image = processing.crop_image_from_coordinates(image, conf['crop_areas'])
                ground_truth, fiber_output = utils.split_image(image)
                ground_truth_stats = analysis.analyze_image(ground_truth)
                fiber_output_stats = analysis.analyze_image(fiber_output)
                
                is_saturated_ground_truth = 1 if ground_truth_stats.get("max_intensity", 0) >= 255 else 0
                is_saturated_fiber_output = 1 if fiber_output_stats.get("max_intensity", 0) >= 255 else 0
                
                coupling_efficiency = None
                if ground_truth_stats.get("total_intensity", 0) > 0:
                    coupling_efficiency = fiber_output_stats.get("total_intensity", 0) / ground_truth_stats.get("total_intensity", 1)
                
                comment = "ND3 (T0.1) filter used; beam repeatition 10hz, 1 bunch; "
                # Save metadata
                meta = {
                    "image_id": str(time.time_ns()), 
                    "capture_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "purpose": experiment.get("purpose", None), 
                    "images_per_sample": experiment.get("images_per_sample", None),  
                    "is_params": experiment.get("is_params", False), 
                    "is_calibration": experiment.get("is_calibration", False), 
                    "is_saturated_ground_truth": is_saturated_ground_truth,
                    "is_saturated_fiber_output": is_saturated_fiber_output,
                    "coupling_efficiency": coupling_efficiency,
                    "ground_truth_img_stat": ground_truth_stats,
                    "fiber_output_img_stats": fiber_output_stats,
                    "image_descriptions": json.dumps({**({"simulation_img": img_size} if experiment.get("include_simulation", False) else {}),
                                                     "ground_truth_img": img_size, "fiber_output_img": img_size}),
                    "image_path": relative_path,  
                    "config_id": config_id,
                    "batch": batch,
                    "comments": comment,
                    "beam_settings": beam_settings,
                }
                ImageMeta.set_metadata(meta)
                cv2.imwrite(image_path, image)
                DB.sql_execute(ImageMeta.to_sql_insert("mmf_dataset_metadata")) 
            
            DMD.free_memory()
            
except KeyboardInterrupt:
    print("\nData collection interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()
finally:
    # Cleanup
    preview_manager.close()
    try:
        param_panel.window.destroy()
    except:
        pass

# Update statistics
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
# Cleanup
# ============================
DB.close()  
DMD.end()
MANAGER.end()

print("\nData collection completed successfully!")
