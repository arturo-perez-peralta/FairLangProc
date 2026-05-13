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
release = '0.1.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]
todo_include_todos = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = [
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "pandas",
    "numpy",
    "scikit-learn",
    "adapter_transformers",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'transformers': ('https://huggingface.co/docs/transformers/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

intersphinx_timeout = 10
intersphinx_cache_limit = 5

def linkcode_resolve(domain, info):
    """Resolve GitHub source links for code objects."""
    if domain != 'py':
        return None
    if not info.get('module'):
        return None
    filename = info['module'].replace('.', '/')
    return f"https://github.com/arturo-perez-peralta/FairLangProc/blob/main/FairLangProc/{filename}.py"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_theme_options = {
    'github_user': 'arturo-perez-peralta',
    'github_repo': 'FairLangProc',
    'github_banner': True,
    'show_related': True,
}

html_context = {
    'display_github': True,
    'github_user': 'arturo-perez-peralta',
    'github_repo': 'FairLangProc',
    'github_banner': True,
}

html_use_index = True
html_split_index = False

htmlhelp_basename = 'FairLangProcdoc'

# -- Options for LaTeX output ----------------------------------------------
root_doc = 'index'
latex_documents = [
    (root_doc, 'fairlangproc.tex', 'FairLangProc Documentation', 'Arturo Perez-Peralta', 'manual'),
]

exclude_patterns += ['**/*.svg']
