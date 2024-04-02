# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
DOCDIR        = ../../docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile deploy clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# New target to copy html files to the docs directory one level up
deploy:
	@echo "Moving built HTML files to the $(DOCDIR) directory..."
	@cp -a "$(BUILDDIR)/html/." "$(DOCDIR)"
	@echo "Deployment of documentation to the $(DOCDIR) directory complete."
	@echo "Removing the build directory..."
	@rm -rf "$(BUILDDIR)"
	@echo "Build directory removed."

