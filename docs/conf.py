# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FairLangProc'
copyright = '2025, Arturo Perez-Peralta'
author = 'Arturo Perez-Peralta'
release = '0.1.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc']
todo_include_todos = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodock_mock_imports = [
    "torch",
    "torchvision",
    "transformers"
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme
html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

# Optional
html_theme_options = {
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_prev_next": False,
}
