from .utils import *
from .evaluation import *
import matplotlib.pyplot as plt
import matplotlib.patches
import tensorflow as tf
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from IPython.display import clear_output


# ------------------- check basic enviornment -------------------
def check_tensorflow_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Success: TensorFlow is using the following GPU(s): {gpus}")
    else:
        print("Failure: TensorFlow did not find any GPUs.")


def check_tensorflow_version():
    print('TensorFlow version:', tf.__version__)
    try:
        import keras
        print('Keras version:', keras.__version__)
    except ImportError:
        print('Keras is not installed as a separate package. Using integrated Keras with TensorFlow.')
    

# ------------------- callback functions for tensorflow fit -------------------
class ImageReconstructionCallback(tf.keras.callbacks.Callback):
    def __init__(self, inputs, save_path: str=None):
        super(ImageReconstructionCallback, self).__init__()
        if isinstance(inputs, tf.data.Dataset):
            for input, label in inputs.take(1):
                self.val_inputs = input[0]
                self.val_labels = label[0]
        else:
            self.val_inputs = inputs[0]
            self.val_labels = inputs[1]
        self.save_path = save_path

    def on_epoch_begin(self, epoch, logs=None):
        plt.clf()
        clear_output(wait=True)
        # Randomly choose one sample from the validation data
        idx = np.random.randint(0, len(self.val_inputs))
        input_image = self.val_inputs[idx:idx+1]  # Keep batch dimension
        ground_truth = self.val_labels[idx]
        reconstructed = self.model.predict(input_image)
        # Plotting
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(input_image[0, ..., 0], cmap='gray', vmin=0, vmax=1)
        plt.title("Input")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(reconstructed[0, ..., 0], cmap='gray', vmin=0, vmax=1)
        plt.title("Reconstructed")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(ground_truth[..., 0], cmap='gray', vmin=0, vmax=1)
        plt.title("Ground Truth")
        plt.axis('off')
        if self.save_path:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_path = f"{self.save_path}/{timestamp}.png"
            plt.savefig(file_path)  # Save the figure with all subplots
            plt.close()  # Close the plot to free up memory
        else:
            plt.show()
            
        if isinstance(input_image, tf.Tensor):
            print(f"input image max pixel: {tf.reduce_max(input_image)}", 
                f"ground truth image max pixel: {tf.reduce_max(ground_truth)}", 
                f"reconstructed image max pixel: {tf.reduce_max(reconstructed)}")
        else:
            print(f"input image max pixel: {input_image.max()}", 
                f"ground truth image max pixel: {ground_truth.max()}", 
                f"reconstructed image max pixel: {reconstructed.max()}")

class PlotPredictionParamsCallback(tf.keras.callbacks.Callback):
    """
    Callback to plot the true and predicted beam parameters on a random validation image
    input: val_images (np.array, (m, n)), val_labels (np.array, (l, 1), normalized), val_beam_image (np.array, (p, q))
    """
    def __init__(self, val_images, val_labels, val_beam_images):
        super().__init__()
        self.val_images = val_images
        self.val_labels = val_labels
        self.val_beam_images = val_beam_images

    def on_epoch_begin(self, epoch, logs=None):
        clear_output(wait=True)
        # Randomly select an image from the validation set
        idx = np.random.randint(0, len(self.val_images))
        image = self.val_beam_images[idx]
        true_label = self.val_labels[idx]
        # Predict the output using the current model state
        pred_label = self.model.predict(self.val_images[idx][np.newaxis, :])[0]
        img_2_params_evaluation(image, true_label, pred_label)
        

class PlotPredictionImageCallback(tf.keras.callbacks.Callback):
    """
    Callback to plot the true and predicted images on a random validation image
    input: x_data (np.array, (n x m), speckle pattern), y_data (np.array, (p x q), beam image)
    """
    def __init__(self, x_data, y_data):
        super(PlotPredictionImageCallback, self).__init__()
        self.x_data = x_data
        self.y_data = y_data

    def on_epoch_end(self, epoch, logs=None, title = ['MMF Speckle Pattern (Input)', 
                                                      'Original Beam Distribution (Ground Truth)',
                                                      'Reconstructed Image (Output)']):
        clear_output(wait=True)
        predictions = self.model.predict(self.x_data[tf.newaxis, ...], verbose=0)
        plt.figure(figsize=(15, 15))
        display_list = [self.x_data.reshape(64, 64), self.y_data.reshape(32, 32),
                        predictions.reshape(32, 32)]

        for i in range(3):  # present the result in a nice visual way
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot. e.g. plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.imshow(display_list[i], cmap='Greys_r')
            plt.axis('off')
        plt.show()


# ------------------- dataset preparation -------------------
def clean_tensor(narray):
    """
    Discard some problematic images based on beam parameters calculation.
    In future, need to develop a better evaluation function (beam_params) to handle this properly?
    """
    labels = list(beam_params(narray, normalize=True).values())
    for i in labels:
        if i >= 1 or i <=0:
            return None, None
    return narray, labels

@print_underscore
def split_dataset(data, labels, proportion=(8, 1, 1)):
    """
    split dataset, Tensorflow only
    dimension: (n, 2, width, hight, channel), (n, beam parameters NO.)
    """
    total = sum(proportion)
    prop_test = proportion[2] / total
    prop_val = proportion[1] / (total - proportion[2]) 
    
    train_val, test, labels_train_val, labels_test = train_test_split(
        data, labels, test_size=prop_test, random_state=42)
    train, val, labels_train, labels_val = train_test_split(
        train_val, labels_train_val, test_size=prop_val, random_state=42)  
    print(f'train set shape: {train.shape}')
    print(f'train label shape: {labels_train.shape}')
    print(f'validation set shape: {val.shape}')
    print(f'validation label shape: {labels_val.shape}')
    print(f'test set shape: {test.shape}')
    print(f'test label shape: {labels_test.shape}')
    return {'x_train' : train, 'label_train' : labels_train, 
            'x_val' : val, 'label_val' : labels_val,
            'x_test' : test, 'label_test' : labels_test} 


def seperate_img(data):
    """
    temp functions for split orignal beam image and speckle pattern for later callback function use
    assume data consists of both beam image and speckle pattern
    """
    new_data = np.transpose(data, (1, 0, 2, 3, 4))
    return new_data[0], new_data[1] # beam image, speckle pattern


# ------------------- experiment logs -------------------
class Logger:
    """
    Create folder and a log file in the specified directory, containing the experiment details (snapshot).
    After training, save the log content in the log file under the log directory.
    """
    def __init__(self, log_dir, model=None, dataset=None, history=None, info=''):
        self.log_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_" + info))
        self.model = model
        self.dataset = dataset
        self.history = history
        self.log_file = os.path.join(self.log_dir, 'log.json')
        self.log_content = {'info' : info,
                            'experiment_date' : datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                            'dataset_info': None,
                            'model_info': None, 
                            'training_info': None}
        self.update()
            
    def update(self):
        if self.dataset is not None:
            self.register_dataset()
        if self.model is not None:
            self.register_model()
        if self.history is not None:
            self.register_training()
            
    def register_extra(self, extra_info):
        self.log_content['extra_info'] = extra_info
            
    def register_dataset(self):
        if isinstance(self.dataset, np.ndarray):
            self.log_content['dataset_info'] = {'dataset_shape': str(self.dataset.shape), 
                                                'dataset_dtype': str(self.dataset.dtype),
                                                'dataset_mean': str(np.mean(self.dataset)), 
                                                'dataset_std': str(np.std(self.dataset)),
                                                'dataset_min': str(np.min(self.dataset)), 
                                                'dataset_max': str(np.max(self.dataset))}

    def register_model(self):
        if isinstance(self.model, tf.keras.models.Model):
            self.log_content['model_info'] = self.tf_model_summary()
        
    def register_training(self):
        os_info = get_system_info()
        if isinstance(self.model, tf.keras.models.Model):
            compiled_info = {
            'loss': self.model.loss,
            'optimizer': type(self.model.optimizer).__name__,
            'optimizer_config': {k:str(v) for k,v in self.model.optimizer.get_config().items()},
            'metrics': [m.name for m in self.model.metrics]
            }
            self.log_content['training_info'] = {'os_info': os_info, 
                                                'compiled_info': compiled_info,
                                                'epoch': len(self.history.epoch),
                                                'training_history': self.history.history
                                                }
            compiled_info['tensorflow_version'] = tf.__version__
            
    def tf_model_summary(self):
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        return summary
        
    def log_parse(self):
        pass
        
    def save(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        with open(self.log_file, 'w') as f:
            json.dump(self.log_content, f, indent=4)
        return self.log_file


# ------------------- intermediate results visualization -------------------
def img_2_params_evaluation(image, true_label, pred_label):
    fig, ax = plt.subplots()
    ax.imshow(image.squeeze(), cmap='gray')  # Display the image

    # Calculate normalized coordinates based on image dimensions
    # These are used for plotting the centroids and ellipses
    true_x = true_label[0] * image.shape[1]
    true_y = true_label[1] * image.shape[0]
    pred_x = pred_label[0] * image.shape[1]
    pred_y = pred_label[1] * image.shape[0]

    # Plot centroids with more professional styling
    ax.plot(true_x, true_y, 'o', markersize=3, markeredgecolor='blue', markerfacecolor='none', label='True Centroid')
    ax.plot(pred_x, pred_y, '^', markersize=3, markeredgecolor='darkred', markerfacecolor='none', label='Predicted Centroid')

    # Plot ellipses with professional style
    true_ellipse = matplotlib.patches.Ellipse((true_x, true_y),
                                              width=true_label[2] * image.shape[1] * 2, 
                                              height=true_label[3] * image.shape[0] * 2,
                                              edgecolor='blue', facecolor='none',
                                              linewidth=1, linestyle='--', label='True Widths')
    ax.add_patch(true_ellipse)
    pred_ellipse = matplotlib.patches.Ellipse((pred_x, pred_y),
                                              width=pred_label[2] * image.shape[1] * 2,
                                              height=pred_label[3] * image.shape[0] * 2,
                                              edgecolor='darkred', facecolor='none',
                                              linewidth=1, linestyle='--', label='Predicted Widths')
    ax.add_patch(pred_ellipse)

    # Set labels and title with normalized axis labels
    ax.set_xlabel('Normalized Horizontal Position')
    ax.set_ylabel('Normalized Vertical Position')
    #ax.set_title('img2params model\'s prediction on a random testset sample', pad=20)

    # Improve the granularity of axis labels
    num_ticks = 10  # More ticks for better granularity
    tick_values = np.linspace(0, 1, num_ticks)
    tick_labels = [f"{x:.1f}" for x in tick_values]
    ax.set_xticks(tick_values * image.shape[1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_values * image.shape[0])
    ax.set_yticklabels(tick_labels)

    plt.legend()
    plt.show()


# ------------------- other functions -------------------


