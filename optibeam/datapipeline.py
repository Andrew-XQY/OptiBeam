import tensorflow as tf
import numpy as np
import pandas as pd
import ast

from PIL import Image
from abc import ABC, abstractmethod
from typing import *
from .utils import get_all_file_paths
from .database import Database


# ----------------- new tf pipeline with prefetch ----------------- 

def load_and_process_image(path):
    image = tf.io.read_file(path) # Read the image file
    # Decode the image to its original depth (assuming the image is grayscale)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32) # float32 type and normalization
    width = tf.shape(image)[1]  # Split the image in half horizontally
    half_width = width // 2
    label = image[:, :half_width]  # left_half
    input = image[:, half_width:]  # right_half
    return input, label

def tf_dataset_prep(data_dirs, func, batch_size, shuffle=True, buffer_size=1024):
    # Create a Dataset from the list of paths
    dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    # Map the processing function to each file path
    dataset = dataset.map(func, num_parallel_calls=tf.data.AUTOTUNE)
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    # Prefetch to improve pipeline performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def datapipeline_conclusion(dataset:tf.data.Dataset, batch_size):
    assert isinstance(dataset, tf.data.Dataset)
    print("total number of batches: ", len(dataset), "with batch size: ", batch_size)
    for left_imgs, right_imgs in dataset.take(1):  
        print(left_imgs.shape, right_imgs.shape)  






# ----------------- old data pipeline ----------------- 
class DataPipeline:
    def __init__(self, df, shape):
        self.df = df
        self.shape = shape
    
    def data_pipeline(self, dim, batch_size=1, is_batch=True):
        batch_x, batch_y = [], []
        while True:  # Loop indefinitely
            for index, row in self.df.iterrows():
                img = Image.open(row['image_path']).convert('L')  # Convert to grayscale
                crop_x = ast.literal_eval(row["speckle_crop_pos"])
                crop_y = ast.literal_eval(row["original_crop_pos"])
                crop_x = tuple(item for subtuple in crop_x for item in subtuple)
                crop_y = tuple(item for subtuple in crop_y for item in subtuple)
                img_x = img.crop(crop_x)  # crop ROI
                img_y = img.crop(crop_y)
                img_x = img_x.resize(dim)   # Resize dimensions
                img_y = img_y.resize(dim)
                res_x = np.expand_dims(np.array(img_x), axis=-1) # Change shape to (256, 256, 1)
                res_y = np.expand_dims(np.array(img_y), axis=-1)
                if is_batch:
                    batch_x.append(np.array(res_x)) 
                    batch_y.append(np.array(res_y)) 
                    if len(batch_x) >= batch_size:  # Yield a batch when batch size is reached
                        batch_x = np.stack(batch_x)
                        batch_y = np.stack(batch_y)
                        yield batch_x.astype('float32') / 255., batch_y.astype('float32') / 255.
                        batch_x, batch_y = [], []
                else:
                    yield res_x.astype('float32') / 255., res_y.astype('float32') / 255.

    def create_tf_dataset(self, batch_list, dim=(256, 256), batch_size=1, is_batch=True):
        return tf.data.Dataset.from_generator(
            generator=lambda: self.data_pipeline(df=self.df[self.df['batch'].isin(batch_list)], dim=dim, batch_size=batch_size),
            output_types=(tf.float32, tf.float32),
            output_shapes=(self.shape, self.shape)
        ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
