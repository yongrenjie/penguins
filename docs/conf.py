# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
# import sys
# sys.path.insert(0, os.path.abspath('../penguins'))


# -- Project information -----------------------------------------------------

project = 'penguins'
copyright = '2020–2021, Jonathan Yong'
author = 'Jonathan Yong'


# -- General configuration ---------------------------------------------------

# https://stackoverflow.com/questions/56336234/
master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive"
]

# Version number
exec(open('../penguins/_version.py').read())
version = __version__

### Matplotlib plot_directive options
# Show source code along with plot
plot_include_source = True
# But don't show a link to the source code
plot_html_show_source_link = False
# Code to run before each plot
plot_pre_code = ""
# Path to run in when generating plots
# Before running this make sure that the submodule is updated!
plot_working_directory = os.path.abspath("../penguins-testdata")
# Make high-quality images...! The low-res PNGs are horrible advertising
plot_formats = ["svg"]
# Don't display links to the source code / images.
plot_html_show_formats = False

### Autodoc options
# Disable type hints in function signatures from being displayed, since this
# information is redundant.
autodoc_typehints='none'

### Intersphinx options
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
}

# Default role for reST. This saves a lot of typing!
default_role = "any"

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# Global reST substitutions
rst_prolog = """
.. |tight_layout| replace:: :func:`plt.tight_layout() <matplotlib.pyplot.tight_layout>`
.. |subplots| replace:: :func:`plt.subplots() <matplotlib.pyplot.subplots>`
.. |figure| replace:: :func:`plt.figure() <matplotlib.pyplot.figure>`
.. |pause| replace:: :func:`plt.pause() <matplotlib.pyplot.pause>`
.. |savefig| replace:: :func:`plt.savefig() <matplotlib.pyplot.savefig>`
.. |mark_inset| replace:: :func:`mark_inset() <mpl_toolkits.axes_grid1.inset_locator.mark_inset>`
.. |show| replace:: :func:`plt.show() <matplotlib.pyplot.show>`
.. |plot| replace:: :meth:`Axes.plot <matplotlib.axes.Axes.plot>`
.. |contour| replace:: :meth:`Axes.contour <matplotlib.axes.Axes.contour>`
.. |legend| replace:: :meth:`Axes.legend <matplotlib.axes.Axes.legend>`
.. |Axes| replace:: :class:`Axes <matplotlib.axes.Axes>`
.. |Figure| replace:: :class:`Figure <matplotlib.figure.Figure>`
.. |Transform| replace:: :class:`Transform <matplotlib.transforms.Transform>`
.. |ndarray| replace:: :class:`ndarray <numpy.ndarray>`
.. |v| replace:: |br| |vspace|
.. |br| raw:: html

   <br />

.. |vspace| raw:: latex

   \\vspace{5mm}

"""

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
html_theme_options = {
    "sidebar_width": "270px",
    "page_width": "1000px",
    "show_relbar_bottom": True,
}
html_favicon = "favicon-32x32.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
