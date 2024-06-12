import platform

# Import modules
from . import utils
from . import evaluation
from . import visualization
from . import training
from . import camera
from . import simulation
from . import database
from . import processing
from . import metadata

# Conditionally import
if platform.system() == 'Windows':
    from . import dmd
    extra_imports = ['dmd']  # ALP4 driver is only available on Windows
else:
    extra_imports = []

# Define what is available to import from the package
__all__ = ['utils', 'evaluation', 'visualization', 'training', 'camera',
           'simulation', 'database', 'processing', 'metadata'] + extra_imports

# Package metadata
__author__ = 'Andrew Xu'
__email__ = 'qiyuanxu95@gmail.com'
__version__ = '0.1.45'
