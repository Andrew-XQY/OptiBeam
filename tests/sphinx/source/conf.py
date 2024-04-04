# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- Path setup --------------------------------------------------------------

import sys, os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
try:
    # optibeam is installed
    import optibeam
except ImportError:
    # optibeam is run from its source checkout
    sys.path.insert(0, os.path.abspath('../../../optibeam'))
    import optibeam

print(optibeam.__author__)
print(optibeam.__version__)


project = optibeam.__package_name__
author = optibeam.__author__
release = optibeam.__version__
copyright = '2024, Andrew Xu'

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
