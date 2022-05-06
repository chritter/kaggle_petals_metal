"""testing hyperparam tuner"""

import tempfile
from unittest import TestCase
from unittest import mock
import tensorflow as tf
from faker import Faker
import pandas as pd
from functools import partial
from pathlib import Path

from kaggle_petals_metal.models.hyperparam_tuning import Tuner

from numpy.random import default_rng


# class test_tuner(TestCase):
#     def test_tuner(self):

#         with tempfile.TemporaryDirectory() as tmpdirname:
#             tuner = Tuner()

#             with mock.patch(
#                 "kaggle_petals_metal.models.hyperparam_tuning.Tuner.get_objective"
#             ) as get_objective:

#                 def mock_objective(
#                     trial, save_trial_results
#                 ):  # pylint: disable=unused-argument

#                     # create fake results table
#                     rng = default_rng(12345)
#                     columns = ["loss", "val_loss", "f1_score", "val_f1_score"]
#                     results = pd.DataFrame(
#                         [[rng.random() for i in range(4)] for j in range(5)],
#                         columns=columns,
#                     )
#                     results["epochs"] = results["trial"] = [1, 2, 3, 4, 5]

#                     save_trial_results(results, Path(tmpdirname))

#                     return 1

#                 get_objective.return_value = partial(
#                     mock_objective, save_trial_results=tuner.save_trial_results
#                 )
#                 tuner.tune(timeout=2)  # 5s trial length
