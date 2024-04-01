#!/bin/bash
# Script to update Sphinx documentation
cd "$(dirname "$0")"

# Run sphinx-apidoc to generate rst files from docstrings
sphinx-apidoc -o docs/source/ ../optibeam/

# Build the HTML documentation
make clean
make html

# Optionally, open the generated documentation in the default browser
open build/html/index.html
