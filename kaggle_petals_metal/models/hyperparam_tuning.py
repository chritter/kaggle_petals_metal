import optuna

# from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState

from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler  # Tree Parzen Estimator (TPE)
from optuna.integration import TFKerasPruningCallback
import os
import math
import gc
import logging
from pathlib import Path
from functools import partial
import tensorflow as tf
import pandas as pd
import click
import plotly_express as px
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_intermediate_values,
)
from optuna.integration import TFKerasPruningCallback

import sys
from sklearn.utils import class_weight
import mlflow
from optuna.integration import MLflowCallback
from git import Repo

sys.path.append("../../")

from optuna.visualization import plot_edf

from kaggle_petals_metal.models.data_generator import DataGenerator
from kaggle_petals_metal.models.train_model import get_model
from kaggle_petals_metal.data.get_class_names import get_class_names


CLASSES = get_class_names()


callback_mlflow = MLflowCallback(
    tracking_uri="http://localhost:5005", metric_name="val_f1_score"
)
repo = Repo(os.getcwd(), search_parent_directories=True)


class TuneObjectives:
    def __init__(self, save_trial_results) -> None:
        self.save_trial_results = save_trial_results

    def _suggest_hyperparams(self, trial):

        # hyperparams
        # lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        lr = 3e-4
        # dropout = trial.suggest_float("dropout", 0.0, 0.8)
        dropout = 0.65
        # size = trial.suggest_categorical("size", ["small"]) #, "medium", "large"])
        size = "small"
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.5)

        return lr, dropout, size, label_smoothing

    def objective_effnet2(self, trial, batch_size, image_size):

        lr, dropout, size, label_smoothing = self._suggest_hyperparams(trial)

        mlflow.log_param("trial", trial.number)
        mlflow.log_param("model_type", "effnet2")
        mlflow.log_param("lr", lr)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("size", size)
        mlflow.log_param("label_smoothing", label_smoothing)
        mlflow.set_tag("commit", repo.head.reference.commit.hexsha)
        mlflow.set_tag("branch", repo.active_branch.name)

        # Clear clutter from previous TensorFlow graphs.
        tf.keras.backend.clear_session()

        ds_train, ds_valid, ds_test = DataGenerator(
            BATCH_SIZE=batch_size,
            IMAGE_SIZE=(image_size, image_size),
            RESIZE=None,
            tpu=False,
        ).get_datasets()

        model = get_model(
            model_type="effnet2",
            hyperparams={
                "lr": lr,
                "dropout": dropout,
                "size": size,
                "label_smoothing": label_smoothing,
            },
            image_size=image_size,
        )

        monitor = "val_f1_score"
        callback_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=0,
            patience=5,
            verbose=1,
            mode="max",
            baseline=None,
            restore_best_weights=False,
        )

        #     callback_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="training/cp-{epoch:04d}.ckpt",
        #                                                      save_weights_only=True,
        #                                                                    monitor='val_f1_score',
        #                                                      verbose=1,  mode='max', save_best_only=True)
        # callback_trial_pruning = TFKerasPruningCallback(trial, monitor)

        compute_steps_per_epoch = lambda x: int(math.ceil(1.0 * x / batch_size))
        steps_per_epoch_tr = compute_steps_per_epoch(12753)
        steps_per_epoch_val = compute_steps_per_epoch(3712)

        # add precomputed weights
        # class_weights = class_weight.compute_class_weight('balanced',
        #                                         np.unique(y_train),
        #                                         y_train)

        history = model.fit(
            ds_train,
            epochs=15,
            validation_data=ds_valid,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch_tr,
            validation_steps=steps_per_epoch_val,
            callbacks=[callback_stopping],  # , callback_trial_pruning],
            shuffle=True,
            verbose=2,
            workers=1,
            use_multiprocessing=False,
        )

        results = pd.DataFrame.from_dict(history.history)
        results["epochs"] = results.index + 1
        best_f1 = results["val_f1_score"].max()

        results["trial"] = trial.number  # get trial number

        best_epoch_vals = results[results["val_f1_score"] == best_f1]
        self.save_trial_results(best_epoch_vals)

        gc.collect()
        del model, ds_train, ds_valid, ds_test
        gc.collect()

        return best_f1

    def get_objective(self, type="effnet2", batch_size=32, image_size=224):
        if type == "effnet2":
            return partial(
                self.objective_effnet2, batch_size=batch_size, image_size=image_size
            )
        else:
            raise ValueError("Unknown objective type: {}".format(type))


class Tuner:
    def __init__(self, save_path, study_name="initial_run") -> None:

        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        sqlite_db = os.path.join(f"sqlite:///{save_path}", "example.db")
        print(sqlite_db)
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            # sampler = optuna.samplers.RandomSampler(seed=42),
            # pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
            # optuna.pruners.SuccessiveHalvingPruner(min_resource='auto',
            #      reduction_factor=4, min_early_stopping_rate=0)
            pruner=optuna.pruners.HyperbandPruner(),
            study_name=study_name,
            storage=sqlite_db,
            load_if_exists=True,
        )

    def save_trial_results(self, results, save_path):
        # if file does not exist write header
        if not os.path.isfile(save_path / "best_vals.csv"):
            results.to_csv(save_path / "best_vals.csv")
        else:
            results.to_csv(save_path / "best_vals.csv", mode="a", header=False)

    def get_objective(self):

        return TuneObjectives(
            save_trial_results=partial(
                self.save_trial_results, save_path=self.save_path
            )
        ).get_objective(type="effnet2", batch_size=32, image_size=224)

    def tune(self, timeout):

        print("Start Optimization.")
        self.study.optimize(
            callback_mlflow.track_in_mlflow()(self.get_objective()),
            # self.get_objective(),
            n_trials=1000000,
            timeout=timeout,
            gc_after_trial=True,
            show_progress_bar=True,
            callbacks=[callback_mlflow],
        )  # timeout after 8hrs: 28800

        show_result(self.study)

        show_best_vals(self.save_path)

        plot_optimization_history(self.study)
        plot_parallel_coordinate(self.study)

        plot_param_importances(self.study)
        plot_intermediate_values(self.study)
        plot_edf(self.study)


def show_result(study):

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def plot_stats_lines(results, var="loss", var_val="val_loss"):
    fig = px.line(
        data_frame=results.groupby("trial").mean().reset_index(),
        x="trial",
        y=var,
        error_y=results.groupby("trial").std().reset_index()[var],
    )
    fig.show()


def show_best_vals(save_path):

    save_path = Path(save_path)
    results_best_epochs = pd.read_csv(save_path / "best_vals.csv")

    plot_stats_lines(results_best_epochs, var="loss", var_val="val_loss")
    plot_stats_lines(results_best_epochs, var="val_f1_score", var_val="val_f1_score")
    plot_stats_lines(results_best_epochs, var="epochs", var_val=None)


@click.command()
@click.argument("timeout", type=int, default=60)
@click.argument("study_name", type=str)  # , default='defaultstudy')
@click.argument("save_path", type=click.Path())  # , default='models/tuning/')
def main(timeout, study_name, save_path):

    save_path = f"{save_path}/{study_name}"

    Tuner(save_path, study_name=study_name).tune(timeout)


if __name__ == "__main__":

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()  # pylint: disable=no-value-for-parameter
