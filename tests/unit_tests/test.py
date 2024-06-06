from conftest import *


dataset_path = '../../ResultsCenter/dataset/2024-06-06/'
dirs = utils.get_all_file_paths(dataset_path)
image_arrays = utils.ImageLoader([]).load(dirs)


visualization.save_as_matplotlib_style_gif(image_arrays, 
                                           frame_rate=30, 
                                           save_path='../../ResultsCenter/dataset/2024-06-06/processed.gif')





















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
















