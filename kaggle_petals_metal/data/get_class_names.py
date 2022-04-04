"""get class names of flowers"""

import json
import pathlib

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()


def get_class_names():
    """
    Reads the classes of flowers and provides names and corresponding indices

    Returns
    -------
    dict
        dictitionary of class names and indices
    """
    with open(CURRENT_FILE_PATH / "classes.json", "r", encoding="utf-8") as jsonfile:
        return json.load(jsonfile)
