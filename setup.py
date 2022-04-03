"""setup.py for installing the package."""

from setuptools import find_packages, setup  # pylint: disable=missing-module-docstring

setup(
    name="kaggle_petals_metal",
    packages=find_packages(include=["kaggle_petals_metal"]),
    version="0.1.0",
    description="Kaggle Competition Petals to the Metal - Flower Classification on TPU",
    author="christian ritter",
    license="MIT",
)
