from conf import *
import pandas as pd

DB = database.SQLiteDB(DATABASE_ROOT)



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
