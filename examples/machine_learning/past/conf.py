"""
Allows local imports of the Optibeam package and sets the working directory to the location of the currently running .py file.
"""

import sys
import os
from contextlib import ContextDecorator
import numpy as np


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
    full_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
    sys.path.insert(0, full_path)
    import optibeam.simulation as simulation
    import optibeam.utils as utils 
    import optibeam.database as database
    import optibeam.processing as processing
    import optibeam.evaluation as evaluation
    import optibeam.training as training
    import optibeam.datapipeline as datapipeline
    # import optibeam.visualization as visualization



# ============================
# Temporary testing functions
# ============================

def apply_intensity_gain(image: np.ndarray, max_gain: float, min_gain: float, scale: float) -> np.ndarray:
    """
    Apply a dynamic gain that boosts low-intensity pixels, avoids overflow,
    and keeps high-intensity pixels minimally affected.

    Parameters:
    - image: np.ndarray
        Input image array with pixel values in the range [0, 255].
    - max_gain: float
        Maximum gain applied to very low-intensity values.
    - min_gain: float
        Minimum gain applied to high-intensity values near 255.
    - scale: float
        Controls how quickly the gain falls off as intensity increases.

    Returns:
    - np.ndarray
        Image array after applying the low-intensity boost, clipped to [0, 255].
    """
    # Convert image to float32 for calculation
    image_float = image.astype(np.float32)
    
    # Normalize pixel values to the range [0, 1]
    normalized_pixels = image_float / 255.0
    
    # Compute dynamic gain:
    # - High values (close to 1) receive a gain near `min_gain`.
    # - Low values (close to 0) receive a gain near `max_gain`.
    dynamic_gain = min_gain + (max_gain - min_gain) * np.exp(-scale * normalized_pixels)
    
    # Apply the dynamic gain
    adjusted_image = image_float * dynamic_gain

    # Clip pixel values to [0, 255] to prevent overflow
    adjusted_image = np.clip(adjusted_image, 0, 255)

    # Convert back to uint8
    return adjusted_image.astype(np.uint8)


