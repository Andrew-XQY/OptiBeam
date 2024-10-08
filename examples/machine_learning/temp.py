import os
script_path = os.path.abspath(__file__)  # Get the absolute path of the current .py file
up_two_levels = os.path.join(os.path.dirname(script_path), '../../')
normalized_path = os.path.normpath(up_two_levels)
os.chdir(normalized_path) # Change the current working directory to the normalized path

from conf import *
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import random
import gc

print(os.getcwd())
training.check_tensorflow_gpu()
training.check_tensorflow_version()
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

DATASET = "2024-08-23"
dev_flag = True
if dev_flag:
    ABS_DIR = f"C:/Users/qiyuanxu/Documents/ResultsCenter/datasets/{DATASET}/"
    SAVE_TO = f'C:/Users/qiyuanxu/Documents/ResultsCenter/result/dev/{DATASET}/' 
    DATABASE_ROOT = ABS_DIR + "db/liverpool.db"
else:
    ABS_DIR = f'../dataset/{DATASET}/'
    SAVE_TO = f'../results/{DATASET}/' 
    DATABASE_ROOT = ABS_DIR + "db/liverpool.db"
    
log_save_path=SAVE_TO + "logs/"
utils.check_and_create_folder(SAVE_TO)
utils.check_and_create_folder(SAVE_TO+'models')
utils.check_and_create_folder(log_save_path)


@tf.keras.utils.register_keras_serializable()
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, 
                             kernel_size=size, 
                             strides=2, 
                             padding='same',
                             kernel_initializer=initializer, 
                             use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

@tf.keras.utils.register_keras_serializable()
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, 
                                    kernel_size=size, 
                                    strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def Autoencoder(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    encoder = [
    downsample(64, 4, apply_batchnorm=False),  # output (batch_size, 128, 128, 64)
    downsample(128, 4),  # output (batch_size, 64, 64, 128)
    downsample(256, 4),  # output (batch_size, 32, 32, 256)
    downsample(512, 4),  # output (batch_size, 16, 16, 512)
    downsample(1024, 4),  # output (batch_size, 8, 8, 1024)
    downsample(1024, 4),  # output (batch_size, 4, 4, 1024)
    ]
    decoder = [
    upsample(1024, 4, apply_dropout=True),  # output (batch_size, 8, 8, 1024)
    upsample(1024, 4, apply_dropout=True),  # output (batch_size, 16, 16, 1024)
    upsample(512, 4),  # output (batch_size, 32, 32, 512)
    upsample(256, 4),  # output (batch_size, 64, 64, 256)
    upsample(128, 4),  # output (batch_size, 128, 128, 128)
    upsample(64, 4),  # output (batch_size, 256, 256, 64)
    ]
    last = tf.keras.layers.Conv2D(input_shape[-1], kernel_size=4, activation='tanh', padding='same')
    x = inputs
    for down in encoder:
        x = down(x)
    for up in decoder:
        x = up(x)
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
    
    # initializer = tf.random_normal_initializer(0., 0.02)
    # # last = tf.keras.layers.Conv2DTranspose(input_shape[-1], 4,
    # #                                         strides=2,
    # #                                         padding='same',
    # #                                         kernel_initializer=initializer,
    # #                                         activation='tanh')  # (batch_size, 256, 256, 1)
    # last = tf.keras.layers.Conv2D(input_shape[-1], kernel_size=4, activation='tanh', padding='same')
    # x = inputs
    # # Downsampling through the model
    # skips = []
    # for down in down_stack:
    #     x = down(x)
    #     skips.append(x
    # skips = reversed(skips[:-1])
    # # Upsampling and establishing the skip connections
    # for up, skip in zip(up_stack, skips):
    #     x = up(x)
    #     #x = tf.keras.layers.Concatenate()([x, skip])
    # x = last(x)
    # return tf.keras.Model(inputs=inputs, outputs=x)


# ------------------------------ dataset preparation -----------------------------------

def load_and_process_image(path):
    image = tf.io.read_file(path) # Read the image file
    # Decode the image to its original depth (assuming the image is grayscale)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32) # float32 type and normalization
    width = tf.shape(image)[1]  # Split the image in half horizontally
    half_width = width // 2
    left_half = image[:, :half_width]
    right_half = image[:, half_width:]
    return left_half, right_half

def tf_dataset_prep(data_dirs, func, batch_size):
    # Create a Dataset from the list of paths
    dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
    # Map the processing function to each file path
    dataset = dataset.map(func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    # Prefetch to improve pipeline performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def datapipeline_conclusion(dataset, batch_size):
    print("total number of batches: ", len(dataset), "with batch size: ", batch_size)
    for left_imgs, right_imgs in dataset.take(1):  
        print(left_imgs.shape, right_imgs.shape)  




batch_size = 4

DB = database.SQLiteDB(DATABASE_ROOT)
# creating training set
sql = """
    SELECT id, batch, image_path
    FROM mmf_dataset_metadata
    WHERE is_calibration = 0 and batch = 2
"""
df = DB.sql_select(sql)
print('Total number of records in the table: ' + str(len(df)))
train_paths = [ABS_DIR+i for i in df["image_path"].to_list()]
train_dataset = tf_dataset_prep(train_paths, load_and_process_image, batch_size)
datapipeline_conclusion(train_dataset, batch_size)

# creating validation set
sql = """
    SELECT id, batch, image_path
    FROM mmf_dataset_metadata
    WHERE is_calibration = 0 and batch = 1
"""
df = DB.sql_select(sql)
print('Total number of records in the table: ' + str(len(df)))
val_paths = [ABS_DIR+i for i in df["image_path"].to_list()]
val_dataset = tf_dataset_prep(val_paths, load_and_process_image, batch_size)
datapipeline_conclusion(val_dataset, batch_size)


