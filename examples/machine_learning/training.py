"""_summary_
Image Reconstruction training using Autoencoder (Optimized from Pix2Pix GAN)
"""
import os
script_path = os.path.abspath(__file__)  # Get the absolute path of the current .py file
up_two_levels = os.path.join(os.path.dirname(script_path), '../../')
normalized_path = os.path.normpath(up_two_levels)
os.chdir(normalized_path) # Change the current working directory to the normalized path
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from conf import *
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
print(os.getcwd())
training.check_tensorflow_gpu()
training.check_tensorflow_version()

DATASET = "2024-08-09"
SAVE_TO = f'../results/{DATASET}/' 
save_path=SAVE_TO + "logs/"
utils.check_and_create_folder(SAVE_TO)
utils.check_and_create_folder(SAVE_TO+'models')
utils.check_and_create_folder(save_path)


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

paths = utils.get_all_file_paths(f'../dataset/{DATASET}/training')
process_funcs = [np.array, utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, 
                 lambda x : (np.expand_dims(x[0], axis=-1), np.expand_dims(x[1], axis=-1))]
loader = utils.ImageLoader(process_funcs)
data = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)
data = np.array(data)
train_X = data[:, 1, :, :, :]
train_Y = data[:, 0, :, :, :]

paths = utils.get_all_file_paths(f'../dataset/{DATASET}/test')
data = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)
data = np.array(data)
val_X = data[:, 1, :, :, :]
val_Y = data[:, 0, :, :, :]

print(f"training input shape:{train_X.shape}\n" + f"training output shape:{train_Y.shape}")
print(f"validation input shape:{val_X.shape}\n" + f"validation output shape:{val_Y.shape}")


# ------------------------------ model training -----------------------------------

shape = train_X.shape[1:]
autoencoder = Autoencoder(shape)
autoencoder.summary()
print(f"model size: {autoencoder.count_params() * 4 / (1024**2)} MB") 

# Initialize early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                  verbose=1, mode='min', restore_best_weights=True)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
autoencoder.compile(optimizer=adam_optimizer, 
                    loss=tf.keras.losses.MeanSquaredError())
history = autoencoder.fit(train_X, train_Y,
                        epochs=80,
                        batch_size=4,
                        shuffle=True,
                        validation_data=(val_X, val_Y),
                        callbacks=[training.ImageReconstructionCallback(val_X, val_Y, save_path), early_stopping],
                        verbose=1  
                        )

# ------------------------------ save models -----------------------------------

autoencoder.save(SAVE_TO+'models/model.keras')
print('model saved!')

# Save the training history
with open(save_path+'training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
