from .utils import *
from .evaluation import *
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import matplotlib.pyplot as plt

# ------------------- callback functions for tensorflow fit -------------------
class PlotPredictionParamsCallback(Callback):
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
        fig, ax = plt.subplots()
        ax.imshow(image.squeeze(), cmap='gray')  # Assuming grayscale images, remove .squeeze() if not applicable
        
        # plot the true and predicted beam parameters
        # Plot true centroids
        ax.plot(true_label[0] * image.shape[1], true_label[1] * image.shape[0],
                'ro', label='True Centroid', markersize=1)
        # Plot predicted centroids
        ax.plot(pred_label[0] * image.shape[1], pred_label[1] * image.shape[0],
                'go', label='Predicted Centroid', markersize=1)
        # Plot ellipses for true widths
        true_ellipse = plt.matplotlib.patches.Ellipse((true_label[0] * image.shape[1], true_label[1] * image.shape[0]),
                                                      width=true_label[2] * image.shape[1] * 2, 
                                                      height=true_label[3] * image.shape[0] * 2,
                                                      edgecolor='r', facecolor='none',
                                                      label='True Widths', linestyle=':')
        ax.add_patch(true_ellipse)
        # Plot ellipses for predicted widths
        pred_ellipse = plt.matplotlib.patches.Ellipse((pred_label[0] * image.shape[1], pred_label[1] * image.shape[0]),
                                                      width=pred_label[2] * image.shape[1] * 2,
                                                      height=pred_label[3] * image.shape[0] * 2,
                                                      edgecolor='g', facecolor='none', 
                                                      label='Predicted Widths', linestyle=':')
        ax.add_patch(pred_ellipse)
        plt.legend()
        plt.show()



class PlotPredictionImageCallback(Callback):
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

def exclude_elements(arr, indices):
    """
    Exclude elements from a NumPy array based on a list of indices.
    """
    mask = np.ones(len(arr), dtype=bool)  # Initially, keep all elements
    mask[indices] = False  # Set to False for indices to be excluded
    return arr[mask]


def get_labels(data) -> np.array:
    temp = []
    for i in data:
        temp.append(list(beam_params(np.squeeze(i[0]), normalize=True).values()))
    return np.array(temp)


def clean_tensors(data):
    """
    Manually discard some problematic images based on beam parameters calculation.
    In future, need to develop a better evaluation function (beam_params) to handle this properly?
    """
    labels = get_labels(data)
    exclude = []

    for index, i in enumerate(labels):
        if [j for j in i if j >= 1 or j <=0]:
            print(index, i)
            exclude.append(index)
            
    print('Before cleaning: ', data.shape, labels.shape)
    data_cleaned = exclude_elements(data, exclude)
    label_cleaned = exclude_elements(labels, exclude)
    print('After cleaning: ', data_cleaned.shape, label_cleaned.shape)
    return data_cleaned, label_cleaned


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
    print("-"*50)
    print(f'train set shape: {train.shape}')
    print(f'train label shape: {labels_train.shape}')
    print(f'validation set shape: {val.shape}')
    print(f'validation label shape: {labels_val.shape}')
    print(f'test set shape: {test.shape}')
    print(f'test label shape: {labels_test.shape}')
    print("-"*50)
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










