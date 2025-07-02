import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "SyntheRela"
copyright = "2025, Martin Jurkovic, Valter Hudovernik"
author = "Martin Jurkovic, Valter Hudovernik"
release = "0.0.4"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
