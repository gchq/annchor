# (C) Crown Copyright GCHQ
"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from pathlib import Path

CONF_FILE_PATH = Path(__file__).absolute()
SOURCE_FOLDER_PATH = CONF_FILE_PATH.parent
DOCS_FOLDER_PATH = SOURCE_FOLDER_PATH.parent
REPO_FOLDER_PATH = DOCS_FOLDER_PATH.parent

sys.path.extend([str(DOCS_FOLDER_PATH), str(SOURCE_FOLDER_PATH), str(REPO_FOLDER_PATH)])


# -- Project information -----------------------------------------------------

project = "ANNchor"
copyright = "2021, GCHQ"
author = "Jonathan H"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "logo_only": True,
}

html_logo = "images/logo_yellow.svg"
