from conftest import *
from ALP4 import *
import time
import numpy as np


# Load the Vialux .dll
DMD = ALP4(version = '4.3')
# Initialize the device
DMD.Initialize()



for i in range(720):
    # Generate the mosaic image
    img = simulation.create_mosaic_image(size=1024)
    M = simulation.compile_transformation_matrix(image=img, radians= i*np.pi/180)
    img = simulation.apply_transformation_matrix(img, M)

    img = dmd.pad_image(img, DMD.nSizeY, DMD.nSizeX)
    # Binary amplitude image (0 or 1)
    bitDepth = 8    
    imgSeq = img.ravel()

    # Allocate the onboard memory for the image sequence
    DMD.SeqAlloc(nbImg = 1, bitDepth = bitDepth)
    # Send the image sequence as a 1D list/array/numpy array
    DMD.SeqPut(imgData = imgSeq)
    # Set image rate to 50 Hz
    DMD.SetTiming(pictureTime = 20000) # in microseconds

    # Run the sequence in an infinite loop
    DMD.Run()
    time.sleep(0.01)



# Stop the sequence display
DMD.Halt()
# Free the sequence from the onboard memory
DMD.FreeSeq()
# De-allocate the device
DMD.Free()

print("Device deallocated, program finished")