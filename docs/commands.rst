Commands
========

Commands to run code.

Download raw image files into data directory

.. code:: bash

    download_kaggle_datay  data/raw


Tuning of model with optuna experiment [experiment] for [seconds] saving the run output to [dir]

.. code:: bash

    tune_hyperparameters [seconds] [experiment] [dir]
    tune_hyperparameters 120 testfast2 models/tuning/
