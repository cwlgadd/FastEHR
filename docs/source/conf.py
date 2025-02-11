# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import sys
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'database'))
sys.path.insert(0, basedir)

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataloader'))
sys.path.insert(0, basedir)

# -- Project information -----------------------------------------------------
project = 'FastEHR'
copyright = '2025, Charles Gadd'
author = 'Charles Gadd'
release = '0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # Supports Google and NumPy-style docstrings
]

autosummary_generate = True  # Automatically generate stub pages for API documentation
autosummary_generate_overwrite = True  # Overwrite existing `.rst` files when regenerating
add_module_names = False  # Prevents showing full module paths

# Ensure method names appear without class names
#autodoc_default_options = {
#    'members': False,
#    'undoc-members': True,
#    'private-members': False,
#    'special-members': "__init__",  
#    'inherited-members': True,
#    'show-inheritance': True,
#    'noindex': True,  # Keep methods indexed but let the event hook modify their names
#}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'classic'
html_static_path = ['_static']
#html_css_files = ["custom.css"]


