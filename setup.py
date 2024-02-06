from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "RELSYNDGB - Relational Synthetic Data Generation Benchmark"
LONG_DESCRIPTION = "RELSYNDGB - A Python package for evaluating synthetic relational datasets"

# Setting up
setup(
    # the name must match the folder name "verysimplemodule"
    name="relsyndgb",
    version=VERSION,
    author="Martin Jurkovic, Valter Hudovernik",
    author_email="martin.jurkovic19@gmail.com, valter.hudovernik@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages("."),
    package_dir={"": "."},
    install_requires=[
        "sdmetrics==0.13.0",
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