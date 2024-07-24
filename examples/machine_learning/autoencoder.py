import os
script_path = os.path.abspath(__file__)  # Get the absolute path of the current .py file
up_two_levels = os.path.join(os.path.dirname(script_path), '../../')
normalized_path = os.path.normpath(up_two_levels)
os.chdir(normalized_path) # Change the current working directory to the normalized path
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from conf import *
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

print(os.getcwd())
print(tf.__version__)
training.check_tensorflow_gpu()

DATASET = "2024-07-23"
SAVE_TO = '../results/'
save_path=SAVE_TO + "evaluations/"
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
        print(input_image.max(), ground_truth.max(), reconstructed.max())

    
class Autoencoder(tf.keras.Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(512, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(512, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(1024, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(1024, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(input_shape[-1], kernel_size=4, activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



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


# class Autoencoder(tf.keras.Model):
#     def __init__(self, input_shape):
#         super(Autoencoder, self).__init__()
#         initializer = tf.random_normal_initializer(0., 0.02)
#         self.inputs = tf.keras.layers.Input(shape=input_shape) # (batch_size, 256, 256, 1)
#         self.encoder = [
#             downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
#             downsample(128, 4),  # (batch_size, 64, 64, 128)
#             downsample(256, 4),  # (batch_size, 32, 32, 256)
#             downsample(512, 4),  # (batch_size, 16, 16, 512)
#             downsample(512, 4),  # (batch_size, 8, 8, 512)
#             downsample(1024, 4),  # (batch_size, 4, 4, 1024)
#         ]
#         self.decoder = [
#             upsample(1024, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
#             upsample(512, 4),  # (batch_size, 8, 8, 512)
#             upsample(512, 4),  # (batch_size, 16, 16, 512)
#             upsample(256, 4),  # (batch_size, 32, 32, 256)
#             upsample(128, 4),  # (batch_size, 64, 64, 128)
#             upsample(64, 4),  # (batch_size, 128, 128, 64)
#         ]
#         last = tf.keras.layers.Conv2DTranspose(input_shape[-1], 4,
#                                             strides=2,
#                                             padding='same',
#                                             kernel_initializer=initializer,
#                                             activation='tanh')   # (batch_size, 256, 256, 1)
#         self.decoder.append(last)

#     def call(self, x):
#         x = self.inputs 
#         skips = []
#         for down in self.encoder:  # connnected in forward sequence
#             x = down(x)
#             skips.append(x)
#         skips = reversed(skips[:-1])
#         # Upsampling and establishing the skip connections
#         for up, skip in zip(self.decoder[:-1], skips):
#             x = up(x)
#             x = tf.keras.layers.Concatenate()([x, skip])  
#         x = self.decoder[-1](x)
#         return tf.keras.Model(inputs=self.inputs, outputs=x)
    






# ------------------------------ dataset preparation -----------------------------------

paths = utils.get_all_file_paths(f'../dataset/{DATASET}/test')[:300]
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
shape = train_X.shape[1:]
autoencoder = Autoencoder(shape)
sample_data = np.random.random((1, *shape))  # Batch size of 1
autoencoder(sample_data)



autoencoder.summary()
print(f"model size: {autoencoder.count_params() * 4 / (1024**2)} MB") 

# adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
autoencoder.compile(optimizer=adam_optimizer, 
                    loss=tf.keras.losses.MeanSquaredError())
history = autoencoder.fit(train_X, train_Y,
                        epochs=80,
                        batch_size=4,
                        shuffle=True,
                        validation_data=(val_X, val_Y),
                        callbacks=[ImageReconstructionCallback(val_X, val_Y)]
                        )

# ------------------------------ save models -----------------------------------
# Save the encoder and decoder
autoencoder.encoder.save(SAVE_TO+'models/encoder.keras')
autoencoder.decoder.save(SAVE_TO+'models/decoder.keras')
print('model saved!')
