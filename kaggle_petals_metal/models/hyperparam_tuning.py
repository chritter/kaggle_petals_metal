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
from optuna.visualization import plot_edf

import sys

sys.path.append("../../")

from kaggle_petals_metal.models.data_generator import DataGenerator
from kaggle_petals_metal.models.train_model import get_model
from kaggle_petals_metal.data.get_class_names import get_class_names


CLASSES = get_class_names()


class TuneObjectives:
    def __init__(self, save_trial_results) -> None:
        self.save_trial_results = save_trial_results

    def objective_effnet2(self, trial, batch_size, image_size):

        # hyperparams
        lr = trial.suggest_float("l1reg", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.8)

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
            hyperparams={"lr": lr, "dropout": dropout},
            image_size=image_size,
        )

        callback_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_score",
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

        compute_steps_per_epoch = lambda x: int(math.ceil(1.0 * x / batch_size))
        steps_per_epoch_tr = compute_steps_per_epoch(12753)
        steps_per_epoch_val = compute_steps_per_epoch(3712)

        history = model.fit(
            ds_train,
            epochs=3,
            validation_data=ds_valid,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch_tr,
            validation_steps=steps_per_epoch_val,
            callbacks=[callback_stopping],
            shuffle=True,
            verbose=0,
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

        study = optuna.create_study(
            direction="maximize",
            # sampler = optuna.samplers.TPESampler,
            # pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
            # optuna.pruners.SuccessiveHalvingPruner(min_resource='auto',
            #      reduction_factor=4, min_early_stopping_rate=0)
            pruner=optuna.pruners.HyperbandPruner(),
            study_name=study_name,
        )

        self.study = study
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

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

        self.study.optimize(
            self.get_objective(), n_trials=1000000, timeout=timeout, gc_after_trial=True
        )  # timeout after 8hrs: 28800

        self.show_result()

        self.show_best_vals()

        plot_optimization_history(self.study)
        plot_parallel_coordinate(self.study)

        plot_param_importances(self.study)
        plot_intermediate_values(self.study)
        plot_edf(self.study)

    def show_result(self):

        pruned_trials = self.study.get_trials(
            deepcopy=False, states=[TrialState.PRUNED]
        )
        complete_trials = self.study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE]
        )

        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def plot_stats_lines(self, results, var="loss", var_val="val_loss"):
        fig = px.line(
            data_frame=results.groupby("trial").mean().reset_index(),
            x="trial",
            y=var,
            error_y=results.groupby("trial").std().reset_index()[var],
        )

    def show_best_vals(self):

        results_best_epochs = pd.read_csv(self.save_path / "best_vals.csv")

        self.plot_stats_lines(results_best_epochs, var="loss", var_val="val_loss")
        self.plot_stats_lines(
            results_best_epochs, var="val_f1_score", var_val="val_f1_score"
        )
        self.plot_stats_lines(results_best_epochs, var="epochs", var_val=None)


@click.command()
@click.argument("timeout", type=int)
@click.argument("study_name", type=str)
@click.argument("save_path", type=click.Path())
def main(save_path, study_name, timeout):

    Tuner(save_path, study_name=study_name).tune(timeout)


if __name__ == "__main__":

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()  # pylint: disable=no-value-for-parameter