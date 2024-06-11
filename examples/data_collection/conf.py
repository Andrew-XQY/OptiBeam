"""
Allows local imports of the Optibeam package and sets the working directory to the location of the currently running .py file.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from contextlib import ContextDecorator

DATABASE_ROOT = "../../ResultsCenter/db/liverpool.db"
DATASET_ROOT = "../../ResultsCenter/dataset/"

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
    

