'''download raw data from Kaggle'''

import pathlib
from kaggle import KaggleApi
import click
from zipfile import ZipFile


FILE_PATH = pathlib.Path(__file__).parent.resolve()

api = KaggleApi()
api.authenticate()

@click.command()
@click.argument('output_path', type=click.Path())
def download_kaggle_data(output_path):
    """
    Downloads the raw data from Kaggle.

    Parameters
    ----------
    output_path : str
        path to raw data directory to host raw kaggle data.
    """

    competition_name = 'tpu-getting-started'
    api.competition_download_files(competition_name, output_path)

    with ZipFile(f"{output_path}/{competition_name}.zip", 'r') as zip_file:
        # Extract all the contents of zip file in different directory
        zip_file.extractall(output_path)
        print("Extraction done.")



if __name__ == '__main__':
    download_kaggle_data() # pylint: disable=no-value-for-parameter
