from setuptools import setup, find_packages

VERSION = "0.0.0"
DESCRIPTION = "RELSYNDGB - Relational Synthetic Data Generation Benchmark"
LONG_DESCRIPTION = "RELSYNDGB - A Python package for evaluating synthetic relational datasets"

# Setting up
setup(
    name="relsyndgb",
    version=VERSION,
    author="Anon",
    author_email="Anon",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages("."),
    package_dir={"": "."},
    install_requires=[
        "sdv>=1.9.0",
        "sdmetrics>=0.13.0",
        "POT==0.9.3"
        ],
    keywords=["python", "relsyndgb", "synthetic data", "relational data", "evaluation", "benchmark"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)