# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile deploy clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# New target to copy html files to the root directory and delete the build directory
deploy:
	@echo "Moving built HTML files to the root of the repository..."
	@cp -a "$(BUILDDIR)/html/." .
	@echo "Deployment of documentation to the repository root complete."
	@echo "Removing the build directory..."
	@rm -rf "$(BUILDDIR)"
	@echo "Build directory removed."

# Optional clean target to remove the copied html files from the root
clean:
	@echo "Cleaning up the HTML files from the root of the repository..."
	# Add commands to remove the HTML files from the root if needed
	@echo "Cleanup complete."
