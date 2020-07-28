from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, "README.md")) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="localvglobal",
    version="0.0",
    description=("Local Vs Global Uncertainty Repo"),
    long_description=long_description,
    author="Tom Grigg",
    author_email="ucabtgr@ucl.ac.uk",
    url="n/a",
    license="MPL-2.0",
    packages=["localvglobal"],
    install_requires=[
        "setuptools>=39.1.0",
        "matplotlib>=2.2.2",
        "torch>=1.0.0",
    ],
    include_package_data=False,
    classifiers=[
        "Development Status :: 0",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
    ],
)
