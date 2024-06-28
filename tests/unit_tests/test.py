from conftest import *




# import numpy as np
# import matplotlib.pyplot as plt

# # Using the generator
# image_generator = simulation.generate_moving_blocks()

# # Visualize the first 5 images
# for _ in range(500):
#     img = next(image_generator)
#     plt.clf()
#     plt.imshow(img, cmap='gray', vmin=0, vmax=255)
#     plt.draw()
#     plt.pause(1)  # Pause for visibility

    
    







import numpy as np
import time
from ALP4 import *

DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))

# Specify the shape of the array, for example (3, 3) for a 3x3 array
array_shape = (256, 256)

while True:
    # Create an array of ones
    img = np.ones(array_shape) * 255
    img = simulation.macro_pixel(img, size=int(1024/img.shape[0]))
    img = dmd.dmd_img_adjustment(img, 1024)
    DMD.display_image(img)
    time.sleep(5)

DMD.end()



























# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

# def plot_image_with_cursor_info(image_path):
#     # Load the image with PIL and convert to grayscale if needed
#     img = Image.open(image_path)
#     if img.mode != 'L':  # Convert to grayscale if not already
#         img = img.convert('L')
#     img = np.array(img)

#     # Create a figure and axis for the plot
#     fig, ax = plt.subplots()
#     # Display the image
#     im = ax.imshow(img, cmap='gray', vmin=0, vmax=255)

#     # Function to be called when the mouse is moved
#     def on_move(event):
#         if event.inaxes == ax:
#             # Get the x and y pixel coordinates
#             x, y = int(event.xdata), int(event.ydata)
#             # Get the pixel value of the image at the given (x, y) location
#             pixel_value = img[y, x]
#             # Update the figure title with pixel coordinates and value
#             ax.set_title(f'Pixel ({x}, {y}): {pixel_value}')
#             fig.canvas.draw_idle()

#     # Connect the motion_notify_event (mouse movement) with the on_move function
#     fig.canvas.mpl_connect('motion_notify_event', on_move)

#     # Show the plot
#     plt.colorbar(im, ax=ax)  # Shows the color scale
#     plt.show()

# # Example usage
# # plot_image_with_cursor_info('C:/Users/qiyuanxu/Documents/DataWarehouse/MMF/procIMGs/processed/35.png')
# plot_image_with_cursor_info('C:/Users/qiyuanxu/Documents/DataWarehouse/MMF/procIMGs_2/processed/991.png')












































# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import time

# while True:
#     # Parameters
#     size = 256  # Size of the matrix
#     max_sigma = size / 8  # Maximum value for sigma_x and sigma_y

#     # Random sigmas
#     sigma_x = np.random.uniform(low=size/64, high=max_sigma)
#     sigma_y = np.random.uniform(low=size/64, high=max_sigma)

#     # Random intensity with condition
#     intensity = np.random.uniform(-10, 10)

#     # Random Rotation
#     angle_degrees = np.random.uniform(0, 360)
#     angle = np.deg2rad(angle_degrees)  # Convert angle to radians for rotation

#     # Random Translation with decaying probability
#     # Define the maximum radius as the canvas diagonal for generality
#     max_radius = size//2 # np.sqrt(2) * (size / 2)
    
#     # Generate a random radius with decreasing probability
#     radius = np.random.exponential(scale=size/4)  # Adjust scale to control decay
#     radius = min(radius, max_radius)  # Limit radius to max_radius
    
#     # Generate a random angle for translation
#     trans_angle = np.random.uniform(0, 2*np.pi)

#     # Convert polar to Cartesian coordinates for the translation
#     dx = radius * np.cos(trans_angle)
#     dy = radius * np.sin(trans_angle)

#     # Create a coordinate grid
#     x = np.linspace(0, size-1, size)
#     y = np.linspace(0, size-1, size)
#     x, y = np.meshgrid(x, y)

#     # Shift the center for rotation
#     x -= size/2
#     y -= size/2

#     # Apply rotation
#     x_new = x * np.cos(angle) - y * np.sin(angle)
#     y_new = x * np.sin(angle) + y * np.cos(angle)

#     # Apply translation
#     x_new += size/2 + dx
#     y_new += size/2 + dy

#     # Check if intensity is negative
#     if intensity < 0:
#         g = np.zeros((size, size))  # Set Gaussian to zero (black canvas)
#     else:
#         # Calculate the 2D Gaussian with different sigma for x and y
#         g = np.exp(-((x_new - size/2)**2 / (2 * sigma_x**2) + (y_new - size/2)**2 / (2 * sigma_y**2)))
#         g *= intensity  # Scale Gaussian by the intensity factor

#     # Plot the transformed Gaussian
#     plt.clf()
#     plt.imshow(g, extent=(0, size, size, 0), cmap='viridis')
#     plt.colorbar()
#     plt.title('Randomly Transformed Anisotropic Gaussian Distribution')
#     plt.draw()
#     plt.pause(1)




# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters for exponential distribution
# scale = 256 / 4  # As you used size/4 in your examples
# samples = 1000  # Number of samples to generate

# # Generate radii using exponential distribution
# radii = np.random.exponential(scale, samples)

# # Plot histogram of the radii
# plt.hist(radii, bins=30, density=True, alpha=0.75, label='Exponential Distribution')
# plt.axvline(scale, color='red', linestyle='dashed', linewidth=1.5, label='Mean = Scale')
# plt.title('Histogram of Radii Generated by Exponential Distribution')
# plt.xlabel('Radius')
# plt.ylabel('Probability Density')
# plt.legend()
# plt.show()







































# def apply_cylindrical_lens_effect(image, focus_axis='horizontal', focus_strength=20.0, defocus_strength=10.0):
#     """
#     Apply a cylindrical lens effect to an image.

#     Parameters:
#     image (numpy.ndarray): Single-channel image in narray form.
#     focus_axis (str): Axis along which the image will be focused. Can be 'horizontal' or 'vertical'.
#     focus_strength (float): Strength of focus (lower values mean more focus, must be >= 0).
#     defocus_strength (float): Strength of defocus (higher values mean more blur, must be > 0).

#     Returns:
#     numpy.ndarray: Transformed image with cylindrical lens effect.
#     """
#     if focus_axis not in ['horizontal', 'vertical']:
#         raise ValueError("focus_axis must be 'horizontal' or 'vertical'")

#     # Adjust sigma values for Gaussian blur based on focus and defocus strengths
#     sigma_x = focus_strength if focus_axis == 'vertical' else defocus_strength
#     sigma_y = defocus_strength if focus_axis == 'vertical' else focus_strength

#     # Apply Gaussian blur to simulate the defocus
#     transformed_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)

#     return transformed_image



























# # Parameters
# size = 256  # Size of the matrix
# sigma_x = size / 32  # Standard deviation of the Gaussian along x-axis
# sigma_y = size / 16   # Standard deviation of the Gaussian along y-axis
# intensity = 10  # Peak intensity of the Gaussian
# angle = np.deg2rad(45)  # Convert angle to radians for rotation
# dx, dy = 30, -30  # Translation distances

# # Create a coordinate grid
# x = np.linspace(0, size-1, size)
# y = np.linspace(0, size-1, size)
# x, y = np.meshgrid(x, y)

# # Shift the center for rotation
# x -= size/2
# y -= size/2

# # Apply rotation
# x_new = x * np.cos(angle) - y * np.sin(angle)
# y_new = x * np.sin(angle) + y * np.cos(angle)

# # Apply translation
# x_new += size/2 + dx
# y_new += size/2 + dy

# # Calculate the 2D Gaussian with different sigma for x and y
# g = np.exp(-((x_new - size/2)**2 / (2 * sigma_x**2) + (y_new - size/2)**2 / (2 * sigma_y**2)))
# g *= intensity  # Scale Gaussian by the intensity factor

# # Plot the transformed Gaussian
# plt.imshow(g, extent=(0, size, size, 0), cmap='viridis')
# plt.colorbar()
# plt.title('Mathematically Transformed Anisotropic Gaussian Distribution')
# plt.show()























# dataset_path = '../../ResultsCenter/dataset/2024-06-06/'
# dirs = utils.get_all_file_paths(dataset_path)
# image_arrays = utils.ImageLoader([]).load(dirs)


# visualization.save_as_matplotlib_style_gif(image_arrays, 
#                                            frame_rate=30, 
#                                            save_path='../../ResultsCenter/dataset/2024-06-06/processed.gif')














# import platform

# # Import modules
# from . import utils
# from . import evaluation
# from . import visualization
# from . import training
# from . import camera
# from . import simulation
# from . import database
# from . import processing
# from . import metadata

# # Conditionally import
# if platform.system() == 'Windows':
#     from . import dmd
#     extra_imports = ['dmd']  # ALP4 driver is only available on Windows
# else:
#     extra_imports = []

# # Define what is available to import from the package
# __all__ = ['utils', 'evaluation', 'visualization', 'training', 'camera',
#            'simulation', 'database', 'processing', 'metadata'] + extra_imports

# # Package metadata
# __author__ = 'Andrew Xu'
# __email__ = 'qiyuanxu95@gmail.com'
# __version__ = '0.1.45'













# import time

# # Example usage
# @utils.timeout(5)  # Timeout after 5 seconds
# def my_function(x):
#     time.sleep(x)  # Simulate long running process
#     return f"Finished in {x} seconds!"

# try:
#     # Function should finish within 5 seconds
#     print(my_function(3))
#     # This call should timeout
#     print(my_function(6))
# except RuntimeError as e:
#     print(e)






# Random Translation with decaying probability
# max_radius = max(self._width, self._height) // 2  # np.sqrt(2) * (size / 2)
# Generate a random radius with decreasing probability
# radius = np.random.exponential(scale=self._width/2)  # Adjust scale to control decay
# radius = min(radius, max_radius)  # Limit radius to max_radius
# radius = max_radius - np.random.exponential(scale=self._width/2)
# # Generate a random angle for translation
# trans_angle = np.random.uniform(0, 2*np.pi)
# # Convert polar to Cartesian coordinates for the translation
# self.dx = radius * np.cos(trans_angle)
# self.dy = radius * np.sin(trans_angle)







