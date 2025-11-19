"""
Allows local imports of the Optibeam package and sets the working directory to the location of the currently running .py file.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime, time
from contextlib import ContextDecorator


name = datetime.datetime.now().strftime('%Y-%m-%d')
DMD_ROTATION_ANGLE = -45 + 2 # DMD rotation angle for image orientation correctionv, minus left, plus right
DATABASE_ROOT = f"../../DataHub/datasets/{name}/db/dataset_meta.db"
DATASET_ROOT = f"../../DataHub/datasets/{name}/dataset/"
# path_to_images = ["../../DataHub/local_images/MMF/procIMGs/processed",
#                   "../../DataHub/local_images/MMF/procIMGs_2/processed"]
path_to_images = ["../../DataHub/local_images/MMF/procIMGs/",
                  "../../DataHub/local_images/MMF/procIMGs_2/"]
minst_path = "../../DataHub/local_images/MNIST_FASHION/t10k-images-idx3-ubyte"

class ChangeDirToFileLocation(ContextDecorator):
    def __enter__(self):
        # Save the current working directory
        self.original_cwd = os.getcwd()
        # Change the working directory to the location of the currently running .py file
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset the working directory back to its original location
        os.chdir(self.original_cwd)


with ChangeDirToFileLocation():
    full_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    sys.path.insert(0, full_path)
    import optibeam.dmd as dmd
    import optibeam.camera as camera
    import optibeam.simulation as simulation
    import optibeam.utils as utils 
    import optibeam.database as database
    import optibeam.processing as processing
    import optibeam.metadata as metadata
    import optibeam.evaluation as evaluation
    import optibeam.visualization as visualization
    import optibeam.analysis as analysis
    import optibeam.basis as basis
    

def read_MNIST_images(filepath):
    with open(filepath, 'rb') as file:
        # Skip the magic number and read dimensions
        magic_number = int.from_bytes(file.read(4), 'big')  # not used here
        num_images = int.from_bytes(file.read(4), 'big')
        rows = int.from_bytes(file.read(4), 'big')
        cols = int.from_bytes(file.read(4), 'big')
        # Read each image into a numpy array
        images = []
        for _ in range(num_images):
            image = np.frombuffer(file.read(rows * cols), dtype=np.uint8)
            image = image.reshape((rows, cols))
            images.append(image)
        return images