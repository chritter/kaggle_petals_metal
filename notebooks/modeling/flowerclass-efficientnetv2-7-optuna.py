# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.021476, "end_time": "2022-04-02T14:16:25.430363", "exception": false, "start_time": "2022-04-02T14:16:25.408887", "status": "completed"} tags=[]
# # Classification with EfficientNetV2 - Hyperparam Search Optuna
#
# ## Goals
#
# * Hyperparam search in dropout space and L1 regularization space
# * Leverage insights from previous notebooks
# * Just train the final layer, no phase 2 modeling
#

# %% papermill={"duration": 10.764615, "end_time": "2022-04-02T14:16:36.214852", "exception": false, "start_time": "2022-04-02T14:16:25.450237", "status": "completed"} tags=[]
import math, re, os
import numpy as np
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


import tensorflow_addons as tfa

print(tf.__version__)
print(tfa.__version__)

from flowerclass_read_tf_ds import get_datasets
import tensorflow_hub as hub
import pandas as pd
import math
import plotly_express as px
import gc

# %% papermill={"duration": 0.200758, "end_time": "2022-04-02T14:16:36.436737", "exception": false, "start_time": "2022-04-02T14:16:36.235979", "status": "completed"} tags=[]
tf.test.gpu_device_name()

# %% [markdown] _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 0.022961, "end_time": "2022-04-02T14:16:36.483480", "exception": false, "start_time": "2022-04-02T14:16:36.460519", "status": "completed"} tags=[]
# # I. Data Loading
#
# * Choose 480x480 as model is fixed: https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2

# %% papermill={"duration": 0.02753, "end_time": "2022-04-02T14:16:36.534772", "exception": false, "start_time": "2022-04-02T14:16:36.507242", "status": "completed"} tags=[]
image_size = 224
batch_size = 64

# %% papermill={"duration": 0.02659, "end_time": "2022-04-02T14:16:36.582748", "exception": false, "start_time": "2022-04-02T14:16:36.556158", "status": "completed"} tags=[]
# #%%debug (50, 480)


# %% [markdown] papermill={"duration": 0.02104, "end_time": "2022-04-02T14:16:36.626667", "exception": false, "start_time": "2022-04-02T14:16:36.605627", "status": "completed"} tags=[]
# # II. Model Setup: EfficientNetV2

# %% papermill={"duration": 0.02744, "end_time": "2022-04-02T14:16:36.675683", "exception": false, "start_time": "2022-04-02T14:16:36.648243", "status": "completed"} tags=[]
# effnet2_base = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2"
# effnet2_base = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2"
effnet2_base = (
    "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2"
)


# %% papermill={"duration": 0.030368, "end_time": "2022-04-02T14:16:36.727081", "exception": false, "start_time": "2022-04-02T14:16:36.696713", "status": "completed"} tags=[]


def get_model(lr, dropout):

    effnet2_tfhub = tf.keras.Sequential(
        [
            # Explicitly define the input shape so the model can be properly
            # loaded by the TFLiteConverter
            tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)),
            hub.KerasLayer(effnet2_base, trainable=False),
            tf.keras.layers.Dropout(rate=dropout),
            tf.keras.layers.Dense(104, activation="softmax"),
        ]
    )
    effnet2_tfhub.build(
        (
            None,
            image_size,
            image_size,
            3,
        )
    )  # This is to be used for subclassed models, which do not know at instantiation time what their inputs look like.

    effnet2_tfhub.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            tfa.metrics.F1Score(num_classes=104, average="macro"),
            tf.keras.metrics.CategoricalAccuracy(
                name="categorical_accuracy", dtype=None
            ),
        ],
    )

    return effnet2_tfhub


# %% [markdown] papermill={"duration": 0.020668, "end_time": "2022-04-02T14:16:36.768903", "exception": false, "start_time": "2022-04-02T14:16:36.748235", "status": "completed"} tags=[]
# Notice large amounts of untrainable params as efficientnetv2 layers are frozen

# %% papermill={"duration": 0.02927, "end_time": "2022-04-02T14:16:36.819033", "exception": false, "start_time": "2022-04-02T14:16:36.789763", "status": "completed"} tags=[]
compute_steps_per_epoch = lambda x: int(math.ceil(1.0 * x / batch_size))
steps_per_epoch_tr = compute_steps_per_epoch(12753)
steps_per_epoch_val = compute_steps_per_epoch(3712)
steps_per_epoch_tr, steps_per_epoch_val

# %% [markdown] papermill={"duration": 0.021564, "end_time": "2022-04-02T14:16:36.861959", "exception": false, "start_time": "2022-04-02T14:16:36.840395", "status": "completed"} tags=[]
# # III. Hyperparam Tuning of Phase 1 with Optuna

# %% papermill={"duration": 0.069291, "end_time": "2022-04-02T14:16:36.952993", "exception": false, "start_time": "2022-04-02T14:16:36.883702", "status": "completed"} tags=[]
import optuna

# from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
from optuna.integration import SkoptSampler

from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler  # Tree Parzen Estimator (TPE)
from optuna.integration import TFKerasPruningCallback

# %% papermill={"duration": 0.031704, "end_time": "2022-04-02T14:16:37.006063", "exception": false, "start_time": "2022-04-02T14:16:36.974359", "status": "completed"} tags=[]
study = optuna.create_study(
    direction="maximize",
    # sampler = optuna.samplers.TPESampler,
    # pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
    # optuna.pruners.SuccessiveHalvingPruner(min_resource='auto',
    #      reduction_factor=4, min_early_stopping_rate=0)
    pruner=optuna.pruners.HyperbandPruner(),
    study_name="initial_run2",
)


# %% papermill={"duration": 0.036822, "end_time": "2022-04-02T14:16:37.065648", "exception": false, "start_time": "2022-04-02T14:16:37.028826", "status": "completed"} tags=[]
def objective(trial):

    # hyperparams
    lr = trial.suggest_float("l1reg", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.8)

    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    ds_train, ds_valid, ds_test = get_datasets(
        BATCH_SIZE=batch_size,
        IMAGE_SIZE=(image_size, image_size),
        RESIZE=None,
        tpu=False,
    )

    model = get_model(lr, dropout)

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

    history = model.fit(
        ds_train,
        epochs=5,
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
    save_trial_results(best_epoch_vals)

    gc.collect()
    del model, ds_train, ds_valid, ds_test
    gc.collect()

    return best_f1


import os


def save_trial_results(df):
    # if file does not exist write header
    if not os.path.isfile("best_vals.csv"):
        df.to_csv("best_vals.csv")
    else:
        df.to_csv("best_vals.csv", mode="a", header=False)


# %% papermill={"duration": 2941.315953, "end_time": "2022-04-02T15:05:38.403362", "exception": false, "start_time": "2022-04-02T14:16:37.087409", "status": "completed"} tags=[]

study.optimize(
    objective, n_trials=10000, timeout=2730, gc_after_trial=True
)  # timeout after 8hrs: 28800


# %% papermill={"duration": 0.052138, "end_time": "2022-04-02T15:05:38.490976", "exception": false, "start_time": "2022-04-02T15:05:38.438838", "status": "completed"} tags=[]


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


def show_best_vals():

    results_best_epochs = pd.read_csv("best_vals.csv")

    plot_stats_lines(results_best_epochs, var="loss", var_val="val_loss")
    plot_stats_lines(results_best_epochs, var="val_f1_score", var_val="val_f1_score")
    plot_stats_lines(results_best_epochs, var="epochs", var_val=None)


def plot_stats_lines(results, var="loss", var_val="val_loss"):
    fig = px.line(
        data_frame=results.groupby("trial").mean().reset_index(),
        x="trial",
        y=var,
        error_y=results.groupby("trial").std().reset_index()[var],
    )

    if var_val is not None:

        fig.add_traces(
            list(
                px.line(
                    data_frame=results.groupby("trial").mean().reset_index(),
                    x="trial",
                    y=var_val,
                    error_y=results.groupby("trial").std().reset_index()[var_val],
                ).select_traces()
            )
        )
        fig.data[1].showlegend = True
        fig.data[1].line.color = "red"
        fig.data[1].name = var_val

    fig.data[0].name = var
    fig.data[0].showlegend = True
    fig.show()


# %% papermill={"duration": 0.045786, "end_time": "2022-04-02T15:05:38.571014", "exception": false, "start_time": "2022-04-02T15:05:38.525228", "status": "completed"} tags=[]
show_result(study)

# %% papermill={"duration": 1.072682, "end_time": "2022-04-02T15:05:39.678772", "exception": false, "start_time": "2022-04-02T15:05:38.606090", "status": "completed"} tags=[]
show_best_vals()

# %% papermill={"duration": 0.040026, "end_time": "2022-04-02T15:05:39.753388", "exception": false, "start_time": "2022-04-02T15:05:39.713362", "status": "completed"} tags=[]
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_intermediate_values,
)
from optuna.visualization import plot_edf

# %% papermill={"duration": 0.047775, "end_time": "2022-04-02T15:05:39.834822", "exception": false, "start_time": "2022-04-02T15:05:39.787047", "status": "completed"} tags=[]
plot_optimization_history(study)

# %% papermill={"duration": 0.066742, "end_time": "2022-04-02T15:05:39.936431", "exception": false, "start_time": "2022-04-02T15:05:39.869689", "status": "completed"} tags=[]
plot_parallel_coordinate(study)

# %% papermill={"duration": 0.315315, "end_time": "2022-04-02T15:05:40.286948", "exception": false, "start_time": "2022-04-02T15:05:39.971633", "status": "completed"} tags=[]
plot_param_importances(study)

# %% papermill={"duration": 0.049415, "end_time": "2022-04-02T15:05:40.372037", "exception": false, "start_time": "2022-04-02T15:05:40.322622", "status": "completed"} tags=[]
plot_intermediate_values(study)

# %% papermill={"duration": 0.051692, "end_time": "2022-04-02T15:05:40.460703", "exception": false, "start_time": "2022-04-02T15:05:40.409011", "status": "completed"} tags=[]
plot_edf(study)
