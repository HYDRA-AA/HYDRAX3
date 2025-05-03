"""
Configuration file for the Sphinx documentation builder.
"""

import os
import sys

# Add the src directory to the path so Sphinx can find the modules
sys.path.insert(0, os.path.abspath('..'))

import src  # noqa: E402

# Project information
project = 'HYDRA Encryption'
copyright = '2025, HYDRA Encryption Project Contributors'
author = 'HYDRA Encryption Project Contributors'
version = src.__version__
release = version

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'myst_parser',  # For Markdown support
]

# Add any paths that contain templates
templates_path = ['_templates']

# Source file extensions
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# List of patterns to exclude from source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
}

# Add any paths that contain custom static files
html_static_path = ['_static']

# Custom sidebar templates
html_sidebars = {
    '**': [
        'relations.html',
        'searchbox.html',
        'navigation.html',
    ]
}

# Output file base name for HTML help builder
htmlhelp_basename = 'HYDRAEncryptionDoc'

# LaTeX output configuration
latex_elements = {}

latex_documents = [
    (master_doc, 'HYDRAEncryption.tex', 'HYDRA Encryption Documentation',
     'HYDRA Encryption Project Contributors', 'manual'),
]

# Grouping the document tree into Texinfo files
texinfo_documents = [
    (master_doc, 'HYDRAEncryption', 'HYDRA Encryption Documentation',
     author, 'HYDRAEncryption', 'A novel encryption algorithm.',
     'Miscellaneous'),
]

# Intersphinx configuration
intersphinx_mapping = {'https://docs.python.org/': None}
