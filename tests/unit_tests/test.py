from conftest import *


dataset_path = '../../ResultsCenter/dataset/2024-06-06/'
dirs = utils.get_all_file_paths(dataset_path)
image_arrays = utils.ImageLoader([]).load(dirs)


visualization.save_as_matplotlib_style_gif(image_arrays, 
                                           frame_rate=30, 
                                           save_path='../../ResultsCenter/dataset/2024-06-06/processed.gif')














# import platform

# # Import modules
# from . import utils
# from . import evaluation
# from . import visualization
# from . import training
# from . import camera
# from . import simulation
# from . import database
# from . import processing
# from . import metadata

# # Conditionally import
# if platform.system() == 'Windows':
#     from . import dmd
#     extra_imports = ['dmd']  # ALP4 driver is only available on Windows
# else:
#     extra_imports = []

# # Define what is available to import from the package
# __all__ = ['utils', 'evaluation', 'visualization', 'training', 'camera',
#            'simulation', 'database', 'processing', 'metadata'] + extra_imports

# # Package metadata
# __author__ = 'Andrew Xu'
# __email__ = 'qiyuanxu95@gmail.com'
# __version__ = '0.1.45'
















# import time

# # Example usage
# @utils.timeout(5)  # Timeout after 5 seconds
# def my_function(x):
#     time.sleep(x)  # Simulate long running process
#     return f"Finished in {x} seconds!"

# try:
#     # Function should finish within 5 seconds
#     print(my_function(3))
#     # This call should timeout
#     print(my_function(6))
# except RuntimeError as e:
#     print(e)
















