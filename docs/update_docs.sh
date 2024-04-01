#!/bin/bash
# Script to update Sphinx documentation
cd "$(dirname "$0")"

# Run sphinx-apidoc to generate rst files from docstrings
sphinx-apidoc -o source/modules/ ../optibeam/

# Build the HTML documentation
make clean
make html && make deploy

# Optionally, open the generated documentation in the default browser
open build/html/index.html
