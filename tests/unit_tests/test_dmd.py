
from ALP4 import *
import time
import numpy as np

def pad_image(img, new_height, new_width, padding_value=0):
    # Calculate the padding sizes
    pad_height = new_height - img.shape[0]
    pad_width = new_width - img.shape[1]
    # Ensure non-negative padding sizes
    if pad_height < 0 or pad_width < 0:
        raise ValueError("New dimensions must be larger than the original image dimensions.")
    # Calculate padding for height and width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    # Apply padding
    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                        mode='constant', constant_values=(padding_value,))
    return padded_img


def create_mosaic_image():
    # Create an empty 1000x1000 array
    image = np.zeros((1000, 1000), dtype=int)
    values = np.linspace(0, 255, 9, dtype=int)
    block_size = 1000 // 3  # Resulting in blocks of approximately 170x170
    for i in range(3):
        for j in range(3):
            value_index = i * 3 + j
            image[i * block_size:(i + 1) * block_size,
                  j * block_size:(j + 1) * block_size] = values[value_index]
    return image


# Load the Vialux .dll
DMD = ALP4(version = '4.3')
# Initialize the device
DMD.Initialize()


# Generate the mosaic image
img = create_mosaic_image()
img = pad_image(img, DMD.nSizeY, DMD.nSizeX)
# DMD.nSizeY,DMD.nSizeX


# Binary amplitude image (0 or 1)
bitDepth = 8    
imgSeq = img.ravel()

# Allocate the onboard memory for the image sequence
DMD.SeqAlloc(nbImg = 1, bitDepth = bitDepth)
# Send the image sequence as a 1D list/array/numpy array
DMD.SeqPut(imgData = imgSeq)
# Set image rate to 50 Hz
DMD.SetTiming(pictureTime = 20000) # in microseconds


print("Image demo started")
# Run the sequence in an infinite loop
DMD.Run()
time.sleep(20)
print("Image demo finished")


# Stop the sequence display
DMD.Halt()
# Free the sequence from the onboard memory
DMD.FreeSeq()
# De-allocate the device
DMD.Free()

print("Device deallocated, program finished")