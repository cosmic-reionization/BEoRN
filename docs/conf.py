# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'BEoRN'
# Automatically update the year
copyright = '%Y'
author = 'BEoRN contributors'
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_nb",
]


## some extensions require additional configuration
# mathjax requires an external dependency path
mathjax_path = (
    "http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)
# autodoc imports the modules to document, but we don't want to install them so we link their path
sys.path.insert(0, str(Path('..', 'src').resolve()))
autodoc_mock_imports = ['numpy', 'scipy', 'matplotlib', 'astropy', 'yaml', 'h5py', 'tools21cm', 'tqdm', 'MAS_library', 'skimage', 'mpi4py']
autodoc_default_options = {
    'member-order': 'groupwise',
    "imported-members": True,
    # include names imported into the package namespace

}
# nb_myst allows to render notebooks as well
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}
nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/cosmic-reionization/beorn",
    "use_repository_button": True,
}
html_static_path = ['_static']
