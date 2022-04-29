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

# %% [markdown]
# # Analysis of EfficientNetV2 Tuning

# %%
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_intermediate_values,
)
from optuna.visualization import plot_edf

from kaggle_petals_metal.models.hyperparam_tuning import show_result, show_best_vals

# %%
study = optuna.create_study(
    study_name="effnetv2_labelsmoothing",
    storage="sqlite:///../../../models/tuning/effnetv2_labelsmoothing/example.db",
    load_if_exists=True,
    direction="maximize",
)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

# %%
df

# %% [markdown]
# Note: params_l1reg is the learning rate (lr).

# %%
show_result(study)

# %%
show_best_vals("../../../models/tuning/testfast2/")

# %%
plot_optimization_history(study)

# %%
plot_parallel_coordinate(study)

# %%
plot_param_importances(study)

# %%

# %%
