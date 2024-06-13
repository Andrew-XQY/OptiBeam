from conf import *
import pandas as pd

def read_MNIST_images(filepath):
    with open(filepath, 'rb') as file:
        # Skip the magic number and read dimensions
        magic_number = int.from_bytes(file.read(4), 'big')  # not used here
        num_images = int.from_bytes(file.read(4), 'big')
        rows = int.from_bytes(file.read(4), 'big')
        cols = int.from_bytes(file.read(4), 'big')

        # Read each image into a numpy array
        images = []
        for _ in range(num_images):
            image = np.frombuffer(file.read(rows * cols), dtype=np.uint8)
            image = image.reshape((rows, cols))
            images.append(image)

        return images

# Example usage
# Replace 'path_to_t10k-images.idx3-ubyte' with the actual file path
minst_path = "../../DataWarehouse/MMF/MNIST_ORG/t10k-images.idx3-ubyte"
images = read_MNIST_images(minst_path)
print(len(images))  
visualization.plot_narray(images[0])






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
