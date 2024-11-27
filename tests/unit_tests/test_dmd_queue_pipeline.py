from conftest import *
import matplotlib.pyplot as plt
import time


def display_images(image_generator):
    """
    Displays images from a generator, pausing for 1 second between each.

    Args:
    image_generator (generator): A generator that yields numpy arrays representing images.
    """
    for image in image_generator:
        
        plt.clf()
        plt.imshow(image) # cmap='gray' for black and white, and 'viridis' for color
        plt.colorbar(label='Pixel value')
        plt.draw()  
        plt.pause(1)  # Pause for a short period, allowing the plot to be updated


# Example processing function
def convert_to_grayscale(image):
    return image.mean(axis=2)

# List of file paths to your images
path_to_images = ["../../ResultsCenter/local_images/MMF/procIMGs/processed",
                    "../../ResultsCenter/local_images/MMF/procIMGs_2/processed"]
paths = utils.get_all_file_paths(path_to_images)
print(len(paths))
file_paths = paths[:5]

# Create the generator
image_gen = simulation.read_local_generator(file_paths, processing_funcs=None)

# Display the images
display_images(image_gen)
print("Done")