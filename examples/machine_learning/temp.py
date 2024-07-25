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

class Autoencoder(tf.keras.Model):
    def __init__(self, input_shape, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.input_shape = input_shape
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
            tf.keras.layers.Conv2D(1024, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
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
            tf.keras.layers.Conv2DTranspose(1024, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
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
            tf.keras.layers.Conv2D(input_shape[-1], kernel_size=4, activation='tanh', padding='same') # sigmoid
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

# ------------------------------ dataset preparation -----------------------------------
DATASET = "2024-07-23"

paths = utils.get_all_file_paths(f'../dataset/{DATASET}/training')[:20]
process_funcs = [np.array, utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, 
                 lambda x : (np.expand_dims(x[0], axis=-1), np.expand_dims(x[1], axis=-1))]
loader = utils.ImageLoader(process_funcs)
data = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)
data = np.array(data)
train_X = data[:, 1, :, :, :]
train_Y = data[:, 0, :, :, :]

paths = utils.get_all_file_paths(f'../dataset/{DATASET}/test')[:10]
process_funcs = [np.array, utils.rgb_to_grayscale, utils.image_normalize, utils.split_image, 
                 lambda x : (np.expand_dims(x[0], axis=-1), np.expand_dims(x[1], axis=-1))]
loader = utils.ImageLoader(process_funcs)
data = utils.add_progress_bar(iterable_arg_index=0)(loader.load_images)(paths)
data = np.array(data)
val_X = data[:, 1, :, :, :]
val_Y = data[:, 0, :, :, :]

print(f"training input shape:{train_X.shape}\n" + f"training output shape:{train_Y.shape}")
print(f"validation input shape:{val_X.shape}\n" + f"validation output shape:{val_Y.shape}")


# ------------------------------ model training -----------------------------------
shape = train_X.shape[1:]
autoencoder = Autoencoder(shape)
sample_data = np.random.random((1, *shape))  # Batch size of 1
autoencoder(sample_data)
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
                        verbose=2
                        )

# ------------------------------ save models -----------------------------------
# Save the encoder and decoder
# autoencoder.encoder.save(SAVE_TO+'models/encoder.keras')
# autoencoder.decoder.save(SAVE_TO+'models/decoder.keras')
autoencoder.save(SAVE_TO+'models/autoencoder.keras')
print('model saved!')

# Save the training history
with open(save_path+'training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
