from conf import *
import json
import pandas as pd



# ============================
# update specific fields
# ============================

DB = database.SQLiteDB(DATABASE_ROOT)
sql = """
    SELECT 
        id, batch, purpose
    FROM 
        mmf_dataset_metadata
    WHERE 
        batch = 6 
"""
df = DB.sql_select(sql)
df['purpose'] = 'fun'
sql = DB.batch_update("mmf_experiment_config", "id", df)
DB.sql_execute(sql, multiple=True)
DB.close()  



























# # Database Initialization
# initialization_db = "examples/data_collection/initialization.py"  
# if not os.path.exists(DATABASE_ROOT):
#     print(f"'{DATABASE_ROOT}' does not exist. Running '{initialization_db}'...")
#     # Run the script
#     subprocess.run(["python", initialization_db], check=True)
    
    
    


# # Clean invalid entrys (where the image number is 0 in mmf_experiment_config)
# sql_clean = """
#     DELETE FROM
#         mmf_experiment_config
#     WHERE 
#         total_images = 0 OR total_images IS NULL;
# """
# DB.sql_execute(sql_clean, multiple=True)









# """
#     This script is the main script for fiber image/speckle pattern data collection
# """
# from conf import *



# loader = utils.ImageLoader()
# file_path = r'C:\Users\qiyuanxu\Documents\DataHub\datasets\2024-12-05\dataset\3\1733390746082512000.png'
# file_path = file_path.replace('\\', '/')
# test_img = loader.load_image(file_path)
# print("load success")
# crop_areas = processing.select_crop_areas_center(test_img, num=2, scale_factor=1.4) 
# print("Crop areas selected: ", crop_areas)
# exit()



































































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
