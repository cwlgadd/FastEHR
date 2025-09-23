# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
#
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
#
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


project = 'FastEHR'
copyright = '2025, Charles Gadd'
author = 'Charles Gadd'
release = '0.0.1'

# HTML theme
html_theme = 'furo'

# Ensure stale stubs are replaced
autosummary_generate = True
autosummary_generate_overwrite = True
#autodoc_default_options = {
#    "members": True,
#    "inherited-members": True,
#    "show-inheritance": True,
#    "undoc-members": True,   # list items even without docstrings
#}
#autodoc_typehints = "description"         # show hints nicely in the doc body
#python_use_unqualified_type_names = True  # cleaner type names

#
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

extensions = [
    # "myst_parser",              # for writing docs in Markdown
    # "sphinx.ext.autodoc",       # pull docstrings
    # "sphinx.ext.autosummary",	 # generate API stub pages
    "sphinx.ext.napoleon",      # Google/NumPy-style docstrings
    # "sphinx.ext.intersphinx",
    # "sphinx.ext.viewcode",
    "sphinx_paramlinks",
    "autoapi.extension",
]

# Point AutoAPI at your package folder (adjust if needed)
autoapi_type = "python"
autoapi_dirs = ["../FastEHR"]
autoapi_add_toctree_entry = True
autoapi_member_order = "bysource"
autoapi_options = ["members", "undoc-members", "show-module-summary"]
autoapi_python_class_content = "both"  # class docstring + __init__ docstring
autoapi_keep_files = False             # For debugging


# For future refactoring
napoleon_google_docstring = False
napoleon_numpy_docstring = False


