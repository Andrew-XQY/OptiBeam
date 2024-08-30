from conftest import *
import cv2
import datetime

manager = camera.MultiBaslerCameraManager()
manager.synchronization()

save_path = "../../ResultsCenter/sync/"
# image =  manager.perodically_scheduled_action_command()
for _ in range(20):
    image = manager.schedule_action_command(int(2000 * 1e6))
    if image is not None:
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        cv2.imwrite(save_path + filename + '.png', image)

manager.end()














# class ImageReconstructionCallback(tf.keras.callbacks.Callback):
#     def __init__(self, model, test_data_generator, num_images=3, dim=(128,128)):
#         super().__init__()
#         self.model = model
#         self.test_data_generator = test_data_generator
#         self.num_images = num_images
#         self.dim = dim

#     def on_epoch_end(self, epoch, logs=None):
#         # Clear the previous figure
#         plt.clf()
#         clear_output(wait=True)
#         x_test, y_test = next(self.test_data_generator)  # Get a batch of test data
#         predicted_y = self.model.predict(x_test)  # Get reconstructed images
#         plt.figure(figsize=(15, 5))   # Plot the original and reconstructed images
#         for i in range(self.num_images):
#             ax = plt.subplot(3, self.num_images, i + 1)
#             plt.imshow(x_test[i].reshape(*self.dim), cmap="gray")
#             plt.title("Input")
#             plt.axis("off")
#             ax = plt.subplot(3, self.num_images, i + 1 + self.num_images)
#             plt.imshow(y_test[i].reshape(*self.dim), cmap="gray")
#             plt.title("Label")
#             plt.axis("off")
#             ax = plt.subplot(3, self.num_images, i + 1 + 2 * self.num_images)
#             plt.imshow(predicted_y[i].reshape(*self.dim), cmap="gray")
#             plt.title("Reconstructed")
#             plt.axis("off")
#         plt.tight_layout()
#         plt.show()






# def save_as_matplotlib_style_gif(image_arrays, frame_rate, save_path, cmap='gray'):
#     """
#     Saves a list of numpy arrays as a GIF, styled to resemble matplotlib plots.

#     Args:
#     image_arrays (iterable of np.array): Iterable of numpy arrays where each array represents an image.
#     frame_rate (float): Number of frames per second.
#     save_path (str): Path to save the GIF file.

#     Returns:
#     None
#     """
#     images = []
#     for img in image_arrays:
#         # Plot the image array with matplotlib to capture the style
#         fig, ax = plt.subplots()
#         ax.imshow(img, aspect='equal', cmap=cmap)  # 'viridis' is a common matplotlib colormap
#         #ax.axis('off')  # Hide axes for a cleaner look

#         # Convert the matplotlib plot to an image array
#         fig.canvas.draw()
#         plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#         plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#         # Append the styled image to the GIF
#         images.append(plot_image)
#         plt.close(fig)
        
#     clip = ImageSequenceClip(images, fps=frame_rate)
#     clip.write_gif(save_path)









# class DataLoader(ABC):
#     def __init__(self, dataset_dirs, batch_size=4, preprocessing_functions = None):
#         """_summary_

#         Args:
#             dataset_dirs (iterable): iterable of file paths to all the dataset samples
#             batch_size (int, optional): yield batch size. Defaults to 4.
#             preprocessing_functions (iterabel): data preprocessing  function, will be appied squential order. Defaults to None.
#         """
#         self.dataset_dirs = dataset_dirs
#         self.batch_size = batch_size
#         self.preprocessing_functions = preprocessing_functions

#     @abstractmethod
#     def default_preprocess_function(self, image):
#         pass
    
#     @abstractmethod
#     def __iter__(self):
#         pass

#     @abstractmethod
#     def __next__(self):
#         pass
    
#     def __len__(self):
#         return len(self.dataset_dirs) // self.batch_size
    
#     def total_len(self):
#         return len(self.dataset_dirs)
























































