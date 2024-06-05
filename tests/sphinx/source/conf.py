# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- Path setup --------------------------------------------------------------

import sys, os
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# List all the modules you want to mock
# MOCK_MODULES = ['numpy', 
#                 'pandas', 
#                 'scipy', 
#                 'matplotlib', 
#                 'matplotlib.pyplot',
#                 'matplotlib.colors',
#                 'matplotlib.cbook',
#                 'matplotlib.figure',
#                 'matplotlib.collections',
#                 'matplotlib.markers',
#                 'matplotlib.patches',
#                 'matplotlib.ticker',
#                 'matplotlib.dates',
#                 'matplotlib.axis',
#                 'matplotlib.scale',
#                 'matplotlib.transforms',
#                 'moviepy',
#                 'moviepy.editor',
#                 'sklearn', 
#                 'sklearn.model_selection',
#                 'skimage',
#                 'skimage.transform',
#                 'scipy.optimize',
#                 'scikit-learn',
#                 'sklearn.decomposition',
#                 'plotly',
#                 'plotly.graph_objects',
#                 'tensorflow', 
#                 'tensorflow.keras',
#                 'tensorflow.keras.callbacks',
#                 'IPython', 
#                 'IPython.display',
#                 'multiprocessing', 
#                 'multiprocess',
#                 'tqdm',
#                 'Pillow',
#                 'seaborn',
#                 'winreg',
#                 ]

# MOCK_MODULES = [
#                 'winreg',
#                 ]
# sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)






# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

try:
    # optibeam is installed
    import optibeam
except ImportError:
    print("__Running from source__")
    # optibeam is run from its source checkout
    full_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
    sys.path.insert(0, full_path)
    import optibeam


project = "OptiBeam"
author = optibeam.__author__
release = "0.1.44"
copyright = f'2024, {author}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    # ... any other extensions need
]


# Exclude patterns: specify the file to be ignored
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'dmd.py', 'dmd']  # Adding 'dmd.py' to be excluded


intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'python': ('https://docs.python.org/3', None),
}


autodoc_members = True
autodoc_member_order = 'groupwise' # bysource, groupwise, alphabetical
autosummary_generate = True
numpydoc_show_class_members = False


source_suffix = ['.rst']
templates_path = ['_templates']
exclude_patterns = ['_build']
language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'pyramid' # sphinx_rtd_theme
html_static_path = ['_static']


# Custom stylesheets
def setup(app):
    app.add_css_file('css/toc.css')
