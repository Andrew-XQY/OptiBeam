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
    original_cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    full_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
    print(full_path)
    sys.path.insert(0, full_path)
    # Mock heavy dependencies
    autodoc_mock_imports = ['numpy', 'pandas', 'scipy', 'moviepy', 'matplotlib', 'tensorflow', 'scikit-learn',
                            'IPython', 'multiprocess', 'tqdm', 'Pillow', 'plotly']
    import optibeam
    os.chdir(original_cwd)

project = "OptiBeam"
author = optibeam.__author__
release = "0.1.39"
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
