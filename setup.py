"""setup.py for installing the package."""

from setuptools import find_packages, setup  # pylint: disable=missing-module-docstring

setup(
    name="kaggle_petals_metal",
    packages=find_packages(include=["kaggle_petals_metal"]),
    version="0.1.0",
    description="Kaggle Competition Petals to the Metal - Flower Classification on TPU",
    author="christian ritter",
    license="MIT",
    entry_points="""
        [console_scripts]
        download_kaggle_data=kaggle_petals_metal.data.download_dataset:download_kaggle_data
        tune_hyperparameters=kaggle_petals_metal.models.hyperparam_tuning:main
    """,
)
