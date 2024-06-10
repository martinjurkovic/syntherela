from setuptools import setup, find_packages

VERSION = "0.0.0"
DESCRIPTION = "SyntheRela - Synthetic Relational Data Generation Benchmark"
LONG_DESCRIPTION = "SyntheRela - A Python package for evaluating synthetic relational datasets"

# Setting up
setup(
    name="syntherela",
    version=VERSION,
    author="Anon",
    author_email="Anon",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages("."),
    package_dir={"": "."},
    install_requires=[
        "sdv>=1.9.0,<2",
        "POT==0.9.3",
        "seaborn==0.13.2",
        "xgboost==2.0.3",
        ],
    keywords=["python", "syntherela", "synthetic data", "relational data", "evaluation", "benchmark"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)