import os
script_path = os.path.abspath(__file__)  # Get the absolute path of the current .py file
up_two_levels = os.path.join(os.path.dirname(script_path), '../../')
normalized_path = os.path.normpath(up_two_levels)
os.chdir(normalized_path) # Change the current working directory to the normalized path

from conf import *
from tqdm import tqdm
import numpy as np 
import datetime, time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, layers, Model, losses

print(os.getcwd())

DATASET = "2024-07-11"
SAVE_TO = '../results/'
save_path=SAVE_TO + "evaluations/"
IMAGE_SHAPE = (256, 256, 1)

# training.check_tensorflow_gpu() # check if the GPU is available
training.check_tensorflow_gpu()


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


class Autoencoder(Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(64, kernel_size=4, strides=2, activation='relu', padding='same'),
            layers.Conv2D(128, kernel_size=4, strides=2, activation='relu', padding='same'),
            layers.Conv2D(256, kernel_size=4, strides=2, activation='relu', padding='same'),
            layers.Conv2D(512, kernel_size=4, strides=2, activation='relu', padding='same'),
            layers.Conv2D(512, kernel_size=4, strides=2, ctivation='relu', padding='same'),
            layers.Conv2D(1024, kernel_size=4, strides=2, activation='relu', padding='same')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(1024, kernel_size=4, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(64, kernel_size=4, strides=2, activation='relu', padding='same'),
            layers.Conv2D(input_shape[-1], kernel_size=4, activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



# ------------------------------ dataset preparation -----------------------------------
paths = utils.get_all_file_paths(roots) 
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

print(f"training input shape:{train_X.shape}\n", f"training output shape:{train_Y.shape}")
print(f"validation input shape:{val_X.shape}\n", f"validation output shape:{val_Y.shape}")


shape = train_X.shape[1:]
autoencoder = Autoencoder(shape)
autoencoder.build((None, *shape))
autoencoder.summary()


adam_optimizer = optimizers.Adam(learning_rate=0.00001)
autoencoder.compile(optimizer=adam_optimizer, loss=losses.MeanSquaredError())
history = autoencoder.fit(train_X, train_Y,
                        epochs=5,
                        batch_size=4,
                        shuffle=True,
                        validation_data=(val_X, val_Y),
                        callbacks=[ImageReconstructionCallback(val_X, val_Y)]
                        )


# ------------------------------ save models -----------------------------------
current_time_seconds = time.time()
timestamp = str(current_time_seconds * 1e9)
model_name = f'{timestamp}.keras'
generator.save(SAVE_TO + f'models/{model_name}')
print('model saved!')




