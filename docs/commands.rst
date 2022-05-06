Commands
========

Commands to run code.

Download raw image files into data directory

.. code:: bash

    download_kaggle_datay  data/raw


start mlflow server for experiment tracking:

.. code:: bash

    cd models/mlflow
    ./start_server.sh


Tuning of model with optuna experiment [experiment] for [seconds] saving the run output to
mlfow and (temporary) optuna study object to models/tuning. The latter allows to
continue the optuna run any time.

.. code:: bash

    # tune_hyperparameters [seconds] [experiment]
    tune_hyperparameters 120 testfast2
