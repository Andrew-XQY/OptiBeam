from conf import *
import pandas as pd


# # dmd calibration pattern generation
# def dmd_calibration_pattern_generation(size):
#     # Create a square image with zeros
#     image = np.zeros((size, size), dtype=np.uint8)
#     # Define the center point
#     center = size // 2
#     image[center, center] = 255  # Set the central point to maximum intensity (white)
#     # Draw boundaries
#     image[0, :] = 255  # Top boundary
#     image[-1, :] = 255  # Bottom boundary
#     image[:, 0] = 255  # Left boundary
#     image[:, -1] = 255  # Right boundary
#     return image

# # Define the size of the image
# image_size = 100

# # Generate the image
# square_image = dmd_calibration_pattern_generation(image_size)

# # Display the image using matplotlib
# plt.imshow(square_image, cmap='gray', interpolation='nearest')
# plt.title("Square Image with Center Point and Boundaries")
# plt.show()













# # Delete images and database according to batch number!!!
# BATCH = 2
# DB = database.SQLiteDB(DATABASE_ROOT)
# select_batch = f"""
#     SELECT image_path FROM mmf_dataset_metadata WHERE batch = {BATCH};
# """
# df = DB.sql_select(select_batch)
# for image in df['image_path']:
#     if os.path.exists(image):
#         os.remove(image)

# tables = ["mmf_dataset_metadata", "mmf_experiment_config"]
# for table in tables:
#     delete_batch = f"""
#         DELETE FROM {table} WHERE batch = {BATCH};
#     """
#     DB.sql_execute(delete_batch)

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
