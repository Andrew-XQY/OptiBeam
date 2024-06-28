from conf import *
# import pandas as pd
# DB = database.SQLiteDB(DATABASE_ROOT)






import numpy as np

def generate_radial_gradient(dim: int=256):
    # Create an empty array of the specified dimensions
    image = np.zeros((dim, dim), dtype=np.float32)
    # Calculate the center coordinates
    center_x, center_y = dim // 2, dim // 2
    # Maximum distance from the center to a corner (radius for decay)
    max_radius = np.sqrt(center_x**2 + center_y**2)
    # Populate the array with intensity values based on radial distance
    for x in range(dim):
        for y in range(dim):
            # Calculate distance from the current pixel to the center
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            # Normalize the distance and calculate intensity
            if distance <= max_radius:
                intensity = 255 * (1 - distance / max_radius)
                image[x, y] = intensity
    return image.astype(np.uint8)

# Example of generating a 256x256 gradient image
gradient_image = generate_radial_gradient(256)
visualization.plot_narray(gradient_image)





















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






# # Delete images and database according to batch number!!!
# BATCH = 2
# 
# select_batch = f"""
#     SELECT image_path FROM mmf_dataset_metadata WHERE batch = {BATCH};
# """
# df = DB.sql_select(select_batch)
# for image in df['image_path']:
#     if os.path.exists(image):
#         os.remove(image)

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
