# __init__.py

class LazyImport:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = __import__(self.module_name, globals(), locals(), [name], 0)
        return getattr(self.module, name)

# Lazy-load modules in package
utils = LazyImport('optibeam.utils')
database = LazyImport('optibeam.database')
evaluation = LazyImport('optibeam.evaluation')
visualization = LazyImport('optibeam.visualization')
training = LazyImport('optibeam.training')
camera = LazyImport('optibeam.camera')
simulation = LazyImport('optibeam.simulation')
processing = LazyImport('optibeam.processing')
metadata = LazyImport('optibeam.metadata')
datapipeline = LazyImport('optibeam.datapipeline')
analysis = LazyImport('optibeam.analysis')

# Optionally, conditionally import platform-specific modules
import platform
if platform.system() == 'Windows':  # ALP4 driver is only available on Windows
    dmd = LazyImport('optibeam.dmd')

# Define what is available to import from the package
__all__ = [
    'utils', 'database', 'evaluation', 'visualization', 'training',
    'camera', 'simulation', 'processing', 'metadata', 'datapipeline',
    'analysis'
]

if platform.system() == 'Windows':
    __all__.append('dmd')

# Package metadata
__author__ = 'Andrew Xu'
__email__ = 'qiyuanxu95@gmail.com'
__version__ = '0.1.47'

