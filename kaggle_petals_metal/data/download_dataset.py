"""download raw data from Kaggle"""

import pathlib
from zipfile import ZipFile
import kaggle
import click


FILE_PATH = pathlib.Path(__file__).parent.resolve()


def _download_kaggle_data(output_path):
    """
    Downloads the raw data from Kaggle.

    Parameters
    ----------
    output_path : str
        path to raw data directory to host raw kaggle data.
    """

    api = kaggle.KaggleApi()
    api.authenticate()

    competition_name = "tpu-getting-started"
    api.competition_download_files(competition_name, output_path)

    with ZipFile(f"{output_path}/{competition_name}.zip", "r") as zip_file:
        # Extract all the contents of zip file in different directory
        zip_file.extractall(output_path)
        print("Extraction done.")


@click.command()
@click.argument("output_path", type=click.Path())
def download_kaggle_data(output_path):

    """
    Downloads the raw data from Kaggle.

    Parameters
    ----------
    output_path : str
        path to raw data directory to host raw kaggle data.
    """

    _download_kaggle_data(output_path)


if __name__ == "__main__":
    download_kaggle_data()  # pylint: disable=no-value-for-parameter
