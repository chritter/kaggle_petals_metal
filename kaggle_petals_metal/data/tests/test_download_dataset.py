"""test download dataset"""

import tempfile
from unittest import TestCase
from unittest import mock
import zipfile
import os
from shutil import make_archive


from kaggle_petals_metal.data.download_dataset import _download_kaggle_data


class test_download_kaggle_data(
    TestCase
):  # pylint: disable=invalid-name, missing-class-docstring
    def test_download_kaggle_data(self):  # pylint: disable=missing-function-docstring

        with mock.patch(
            "kaggle_petals_metal.data.download_dataset.kaggle.KaggleApi"
        ) as mock_api:
            # mock_api.return_value.authenticate.return_value = None
            # mock_api.return_value.competition_download_files.return_value = None
            with tempfile.TemporaryDirectory() as tmpdirname:

                tmp_zip_file = os.path.join(tmpdirname, "tpu-getting-started")

                make_archive(tmp_zip_file, "zip", tmpdirname)

                _download_kaggle_data(tmpdirname)

                mock_api.return_value.authenticate.assert_called_once()
                mock_api.return_value.competition_download_files.assert_called_once()
