import os
script_path = os.path.abspath(__file__)  # Get the absolute path of the current .py file
up_two_levels = os.path.join(os.path.dirname(script_path), '../../')
normalized_path = os.path.normpath(up_two_levels)
os.chdir(normalized_path) # Change the current working directory to the normalized path

import model
from conf import *
from tqdm import tqdm
import cv2
import tensorflow as tf
import numpy as np 
import datetime, time

print(os.getcwd())

DATASET = "2024-07-11"
SAVE_TO = '../results/'
IMAGE_SHAPE = (256, 256, 1)

# training.check_tensorflow_gpu() # check if the GPU is available
training.check_tensorflow_gpu()

# ------------------------------ prepare datasets ------------------------------
paths = utils.get_all_file_paths("../dataset/2024-07-11/training") 

process_funcs = [np.array, utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, 
                 lambda x : (np.expand_dims(x[0], axis=-1), np.expand_dims(x[1], axis=-1))]

loader = utils.ImageLoader(process_funcs)
data = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)



data_cleaned, label_cleaned = [], []

for i in range(len(data)):
    d, l = training.clean_tensor(data[i][0].squeeze())
    if d is not None and l is not None:
        data_cleaned.append(data[i]) 
        label_cleaned.append(l)

data_cleaned, label_cleaned = np.array(data_cleaned), np.array(label_cleaned)
print(len(data_cleaned), len(label_cleaned))
print(data_cleaned.shape)



