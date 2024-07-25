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

SAVE_TO = '../results/'
save_path=SAVE_TO + "logs/"
IMAGE_SHAPE = (256, 256, 1)

class ImageReconstructionCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_inputs, val_labels):
        super(ImageReconstructionCallback, self).__init__()
        self.val_inputs = val_inputs
        self.val_labels = val_labels

    def on_epoch_begin(self, epoch, logs=None):
        # Randomly choose one sample from the validation data
        idx = np.random.randint(0, len(self.val_inputs))
        input_image = self.val_inputs[idx:idx+1]  # Keep batch dimension
        ground_truth = self.val_labels[idx]
        reconstructed = self.model.predict(input_image)
        # Plotting
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(input_image[0, ..., 0], cmap='gray')
        plt.title("Input")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(reconstructed[0, ..., 0], cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(ground_truth[..., 0], cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = f"{save_path}/{timestamp}.png"
        plt.savefig(file_path)  # Save the figure with all subplots
        plt.close()  # Close the plot to free up memory
        print(f"input image max pixel: {input_image.max()}", 
              f"ground truth image max pixel: {ground_truth.max()}", 
              f"reconstructed image max pixel: {reconstructed.max()}"
              )


@tf.keras.utils.register_keras_serializable()
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

@tf.keras.utils.register_keras_serializable()
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def Autoencoder(input_shape=[256, 256, 1]):
    inputs = tf.keras.layers.Input(shape=input_shape)
    down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(1024, 4),  # (batch_size, 8, 8, 512)
    downsample(1024, 4),  # (batch_size, 4, 4, 512)
    ]
    up_stack = [
    upsample(1024, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(1024, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 256, 256, 1)
    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        #x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


# ------------------------------ dataset preparation -----------------------------------
DATASET = "2024-07-11"


paths = utils.get_all_file_paths(f'../dataset/{DATASET}/test')[:50]
process_funcs = [np.array, utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, 
                 lambda x : (np.expand_dims(x[0], axis=-1), np.expand_dims(x[1], axis=-1))]
loader = utils.ImageLoader(process_funcs)
data = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)

splite = 0.9
train = np.array(data[:int(len(data)*splite)])
val = np.array(data[int(len(data)*splite):])
train_X = train[:, 1, :, :, :]
train_Y = train[:, 0, :, :, :]
val_X = val[:, 1, :, :, :]
val_Y = val[:, 0, :, :, :]

# paths = utils.get_all_file_paths(f'../dataset/{DATASET}/training') 
# process_funcs = [np.array, utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, 
#                  lambda x : (np.expand_dims(x[0], axis=-1), np.expand_dims(x[1], axis=-1))]
# loader = utils.ImageLoader(process_funcs)
# data = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)
# data = np.array(data)
# train_X = data[:, 1, :, :, :]
# train_Y = data[:, 0, :, :, :]

# paths = utils.get_all_file_paths(f'../dataset/{DATASET}/test') 
# process_funcs = [np.array, utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, 
#                  lambda x : (np.expand_dims(x[0], axis=-1), np.expand_dims(x[1], axis=-1))]
# loader = utils.ImageLoader(process_funcs)
# data = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)
# data = np.array(data)
# val_X = data[:, 1, :, :, :]
# val_Y = data[:, 0, :, :, :]

print(f"training input shape:{train_X.shape}\n" + f"training output shape:{train_Y.shape}")
print(f"validation input shape:{val_X.shape}\n" + f"validation output shape:{val_Y.shape}")


# ------------------------------ model training -----------------------------------
# shape = train_X.shape[1:]
# autoencoder = Autoencoder(shape)
# sample_data = np.random.random((1, *shape))  # Batch size of 1
# autoencoder(sample_data)

autoencoder = Autoencoder()
autoencoder.summary()
print(f"model size: {autoencoder.count_params() * 4 / (1024**2)} MB") 


# Initialize early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                  verbose=1, mode='min', restore_best_weights=True)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
autoencoder.compile(optimizer=adam_optimizer, 
                    loss=tf.keras.losses.MeanSquaredError())
history = autoencoder.fit(train_X, train_Y,
                        epochs=1,
                        batch_size=4,
                        shuffle=True,
                        validation_data=(val_X, val_Y),
                        callbacks=[ImageReconstructionCallback(val_X, val_Y), early_stopping],
                        verbose=1  
                        )

# ------------------------------ save models -----------------------------------
# Save the encoder and decoder
# autoencoder.encoder.save(SAVE_TO+'models/encoder.keras')
# autoencoder.decoder.save(SAVE_TO+'models/decoder.keras')

autoencoder.save(SAVE_TO+'models/autoencoder.keras')
# autoencoder.save(SAVE_TO+'models/autoencoder.keras')
print('model saved!')


# Save the training history
with open(save_path+'training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)




