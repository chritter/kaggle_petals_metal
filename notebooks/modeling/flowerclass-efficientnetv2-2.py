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

# %% [markdown] papermill={"duration": 0.042114, "end_time": "2022-03-05T13:16:00.272480", "exception": false, "start_time": "2022-03-05T13:16:00.230366", "status": "completed"} tags=[]
# # Classification with EfficientNetV2
#
# * Original Google Repo: https://github.com/google/automl/tree/master/efficientnetv2
# * Paper published 2021

# %% papermill={"duration": 8.642366, "end_time": "2022-03-05T13:16:08.955735", "exception": false, "start_time": "2022-03-05T13:16:00.313369", "status": "completed"} tags=[]
import math, re, os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

print(tf.__version__)
print(tfa.__version__)

from flowerclass_read_tf_ds import get_datasets
import tensorflow_hub as hub
import pandas as pd
import math
import plotly_express as px

# %% papermill={"duration": 2.028115, "end_time": "2022-03-05T13:16:11.024688", "exception": false, "start_time": "2022-03-05T13:16:08.996573", "status": "completed"} tags=[]
tf.test.gpu_device_name()

# %% [markdown] _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 0.039961, "end_time": "2022-03-05T13:16:11.104977", "exception": false, "start_time": "2022-03-05T13:16:11.065016", "status": "completed"} tags=[]
# # I. Data Loading
#
# * Choose 480x480 as model is fixed: https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2

# %% papermill={"duration": 0.045615, "end_time": "2022-03-05T13:16:11.190781", "exception": false, "start_time": "2022-03-05T13:16:11.145166", "status": "completed"} tags=[]
image_size = 224
batch_size = 64

# %% papermill={"duration": 0.681329, "end_time": "2022-03-05T13:16:11.912245", "exception": false, "start_time": "2022-03-05T13:16:11.230916", "status": "completed"} tags=[]
# #%%debug (50, 480)
ds_train, ds_valid, ds_test = get_datasets(
    BATCH_SIZE=batch_size, IMAGE_SIZE=(image_size, image_size), RESIZE=None, tpu=False
)

# %% [markdown] papermill={"duration": 0.040612, "end_time": "2022-03-05T13:16:11.993890", "exception": false, "start_time": "2022-03-05T13:16:11.953278", "status": "completed"} tags=[]
# # II. Model Setup: EfficientNetV2

# %% papermill={"duration": 0.047334, "end_time": "2022-03-05T13:16:12.081638", "exception": false, "start_time": "2022-03-05T13:16:12.034304", "status": "completed"} tags=[]
# effnet2_base = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2"
# effnet2_base = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2"
effnet2_base = (
    "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2"
)

# %% papermill={"duration": 0.048564, "end_time": "2022-03-05T13:16:12.170207", "exception": false, "start_time": "2022-03-05T13:16:12.121643", "status": "completed"} tags=[]
hub.KerasLayer

# %% papermill={"duration": 13.704681, "end_time": "2022-03-05T13:16:25.915284", "exception": false, "start_time": "2022-03-05T13:16:12.210603", "status": "completed"} tags=[]

effnet2_tfhub = tf.keras.Sequential(
    [
        # Explicitly define the input shape so the model can be properly
        # loaded by the TFLiteConverter
        tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)),
        hub.KerasLayer(effnet2_base, trainable=False),
        tf.keras.layers.Dropout(rate=0.2),
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


effnet2_tfhub.summary()

# %% [markdown] papermill={"duration": 0.04171, "end_time": "2022-03-05T13:16:25.999205", "exception": false, "start_time": "2022-03-05T13:16:25.957495", "status": "completed"} tags=[]
# Notice large amounts of untrainable params as efficientnetv2 layers are frozen

# %% papermill={"duration": 0.048764, "end_time": "2022-03-05T13:16:26.090386", "exception": false, "start_time": "2022-03-05T13:16:26.041622", "status": "completed"} tags=[]
effnet2_tfhub.layers

# %% papermill={"duration": 0.051243, "end_time": "2022-03-05T13:16:26.183341", "exception": false, "start_time": "2022-03-05T13:16:26.132098", "status": "completed"} tags=[]
layer = effnet2_tfhub.layers[0]
print("weights:", len(layer.weights))
print("trainable_weights:", len(layer.trainable_weights))
print("non_trainable_weights:", len(layer.non_trainable_weights))

# %% papermill={"duration": 0.049798, "end_time": "2022-03-05T13:16:26.274711", "exception": false, "start_time": "2022-03-05T13:16:26.224913", "status": "completed"} tags=[]
layer.weights[0].shape


# %% papermill={"duration": 0.04948, "end_time": "2022-03-05T13:16:26.366532", "exception": false, "start_time": "2022-03-05T13:16:26.317052", "status": "completed"} tags=[]
layer.trainable

# %% [markdown] papermill={"duration": 0.043095, "end_time": "2022-03-05T13:16:26.453770", "exception": false, "start_time": "2022-03-05T13:16:26.410675", "status": "completed"} tags=[]
# Why?

# %% [markdown] papermill={"duration": 0.04289, "end_time": "2022-03-05T13:16:26.538937", "exception": false, "start_time": "2022-03-05T13:16:26.496047", "status": "completed"} tags=[]
# # III. Training

# %% [markdown] papermill={"duration": 0.043507, "end_time": "2022-03-05T13:16:26.624925", "exception": false, "start_time": "2022-03-05T13:16:26.581418", "status": "completed"} tags=[]
# Keras Transfer Learning: https://keras.io/guides/transfer_learning/

# %% [markdown] papermill={"duration": 0.041917, "end_time": "2022-03-05T13:16:26.709196", "exception": false, "start_time": "2022-03-05T13:16:26.667279", "status": "completed"} tags=[]
# # IIIa) Phase I: Train Top Layer (frozen layers)

# %% [markdown] papermill={"duration": 0.042142, "end_time": "2022-03-05T13:16:26.793736", "exception": false, "start_time": "2022-03-05T13:16:26.751594", "status": "completed"} tags=[]
# ### Optimize Training for Compute Infrastructure

# %% papermill={"duration": 0.065449, "end_time": "2022-03-05T13:16:26.901303", "exception": false, "start_time": "2022-03-05T13:16:26.835854", "status": "completed"} tags=[]
effnet2_tfhub.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=[
        tfa.metrics.F1Score(num_classes=104, average="macro"),
        tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None),
    ],
)

# %% [markdown] papermill={"duration": 0.042092, "end_time": "2022-03-05T13:16:26.985537", "exception": false, "start_time": "2022-03-05T13:16:26.943445", "status": "completed"} tags=[]
# * batchsize:4 with 512 resized to 480px OOM
#
# * `effnet2L_tfhub.fit(ds_train, epochs=1, validation_data=ds_valid, batch_size=batch_size, steps_per_epoch=1)`
#
# #### EfficientNetV2 Large
#
# * try batchsize 4, 8, 16 and image size  224, 331 (without resizing for now)
# * bs/image size (no resize)
#     * 8/224: pass
#     * 16/224 pass
#     * 32/224 pass
#     * 64/224 pass
#     * 128/224 pass
# * try 331 (second largest size of images available) with efficientetV2 small
#     * 16/331: OOM
#     * 8/331: OOM
#
# * Test with optimal 480x480 input:
#
#    * 8/448 (resized 480): OOM
#    * 8/224 (resized 480): OOM
#    * 2/224 (resized 480): OOM
#    > Resizing to the optimal 480x480 image size not possible with EfficientNetV2 Large due to OOM
#
#
#
# #### EfficientNetV2 Medium
#
# * Test with optimal 480x480 input:
#
#    * 2/224 (resized 480): OOM
#
# #### EfficientNetV2 Small
#
# * Test with optimal 384 x 384: OOM
#
# > All 3 model types, small, medium, large cannot be used with their optimal resolution.
# >
#
#
#

# %% papermill={"duration": 0.050825, "end_time": "2022-03-05T13:16:27.079003", "exception": false, "start_time": "2022-03-05T13:16:27.028178", "status": "completed"} tags=[]
compute_steps_per_epoch = lambda x: int(math.ceil(1.0 * x / batch_size))
steps_per_epoch_tr = compute_steps_per_epoch(12753)
steps_per_epoch_val = compute_steps_per_epoch(3712)
steps_per_epoch_tr, steps_per_epoch_val

# %% papermill={"duration": 622.680749, "end_time": "2022-03-05T13:26:49.802062", "exception": false, "start_time": "2022-03-05T13:16:27.121313", "status": "completed"} tags=[]
callback_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_f1_score",
    min_delta=0,
    patience=5,
    verbose=1,
    mode="max",
    baseline=None,
    restore_best_weights=False,
)
callback_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="training/cp-{epoch:04d}.ckpt",
    save_weights_only=True,
    monitor="val_f1_score",
    verbose=1,
    mode="max",
    save_best_only=True,
)

history = effnet2_tfhub.fit(
    ds_train,
    epochs=40,
    validation_data=ds_valid,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch_tr,
    validation_steps=steps_per_epoch_val,
    callbacks=[callback_stopping, callback_model_checkpoint],
    shuffle=True,
)

# %% papermill={"duration": 23.127791, "end_time": "2022-03-05T13:27:14.072204", "exception": false, "start_time": "2022-03-05T13:26:50.944413", "status": "completed"} tags=[]
effnet2_tfhub.save("saved_model/my_model_phase1")

# %% papermill={"duration": 1.150127, "end_time": "2022-03-05T13:27:16.414597", "exception": false, "start_time": "2022-03-05T13:27:15.264470", "status": "completed"} tags=[]
results_tr = pd.DataFrame.from_dict(history.history)
results_tr["epochs"] = results_tr.index + 1
results_tr.head()

results_to_plot = results_tr.melt(id_vars="epochs")
results_to_plot.head()

# %% papermill={"duration": 1.126786, "end_time": "2022-03-05T13:27:18.681275", "exception": false, "start_time": "2022-03-05T13:27:17.554489", "status": "completed"} tags=[]
results_to_plot["variable"].unique()

# %% papermill={"duration": 1.909043, "end_time": "2022-03-05T13:27:21.681510", "exception": false, "start_time": "2022-03-05T13:27:19.772467", "status": "completed"} tags=[]
px.line(
    data_frame=results_to_plot[results_to_plot.variable.isin(["loss", "val_loss"])],
    x="epochs",
    y="value",
    color="variable",
)

# %% papermill={"duration": 1.158923, "end_time": "2022-03-05T13:27:23.937109", "exception": false, "start_time": "2022-03-05T13:27:22.778186", "status": "completed"} tags=[]
px.line(
    data_frame=results_to_plot[
        results_to_plot.variable.isin(["f1_score", "val_f1_score"])
    ],
    x="epochs",
    y="value",
    color="variable",
)

# %% papermill={"duration": 1.121495, "end_time": "2022-03-05T13:27:26.416602", "exception": false, "start_time": "2022-03-05T13:27:25.295107", "status": "completed"} tags=[]
best_phase1_f1 = results_tr["val_f1_score"].max()
best_phase1_epoch = results_tr.loc[
    results_tr["val_f1_score"] == best_phase1_f1, "epochs"
].values[0]


# %% papermill={"duration": 1.12131, "end_time": "2022-03-05T13:27:28.675670", "exception": false, "start_time": "2022-03-05T13:27:27.554360", "status": "completed"} tags=[]
best_phase1_f1, best_phase1_epoch

# %% [markdown] papermill={"duration": 1.11766, "end_time": "2022-03-05T13:27:30.901099", "exception": false, "start_time": "2022-03-05T13:27:29.783439", "status": "completed"} tags=[]
# ## IIIb) Phase II: Unfreeze and FineTuning
#
# Unfreeze weights, try fine tuning whole network

# %% papermill={"duration": 1.103394, "end_time": "2022-03-05T13:27:33.104352", "exception": false, "start_time": "2022-03-05T13:27:32.000958", "status": "completed"} tags=[]
effnet2_tfhub.trainable = True

# %% papermill={"duration": 1.151235, "end_time": "2022-03-05T13:27:35.359507", "exception": false, "start_time": "2022-03-05T13:27:34.208272", "status": "completed"} tags=[]
effnet2_tfhub.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=[
        tfa.metrics.F1Score(num_classes=104, average="macro"),
        tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None),
    ],
)

# %% papermill={"duration": 1121.277755, "end_time": "2022-03-05T13:46:18.042631", "exception": false, "start_time": "2022-03-05T13:27:36.764876", "status": "completed"} tags=[]
callback_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_f1_score",
    min_delta=0,
    patience=5,
    verbose=1,
    mode="max",
    baseline=None,
    restore_best_weights=False,
)
callback_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="training2/cp-{epoch:04d}.ckpt",
    save_weights_only=True,
    monitor="val_f1_score",
    verbose=1,
    mode="max",
    save_best_only=True,
)

history = effnet2_tfhub.fit(
    ds_train,
    epochs=10,
    validation_data=ds_valid,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch_tr,
    validation_steps=steps_per_epoch_val,
    callbacks=[callback_stopping, callback_model_checkpoint],
    shuffle=True,
)

# %% papermill={"duration": 27.726274, "end_time": "2022-03-05T13:46:47.585664", "exception": false, "start_time": "2022-03-05T13:46:19.859390", "status": "completed"} tags=[]
effnet2_tfhub.save("saved_model/my_model_phase2")

# %% papermill={"duration": 2.043066, "end_time": "2022-03-05T13:46:51.366343", "exception": false, "start_time": "2022-03-05T13:46:49.323277", "status": "completed"} tags=[]
results_tr = pd.DataFrame.from_dict(history.history)
results_tr["epochs"] = results_tr.index + 1
results_tr.head()

results_to_plot = results_tr.melt(id_vars="epochs")
results_to_plot.head()

# %% papermill={"duration": 1.810605, "end_time": "2022-03-05T13:46:54.985185", "exception": false, "start_time": "2022-03-05T13:46:53.174580", "status": "completed"} tags=[]
px.line(
    data_frame=results_to_plot[results_to_plot.variable.isin(["loss", "val_loss"])],
    x="epochs",
    y="value",
    color="variable",
)

# %% papermill={"duration": 1.819577, "end_time": "2022-03-05T13:46:58.555246", "exception": false, "start_time": "2022-03-05T13:46:56.735669", "status": "completed"} tags=[]
px.line(
    data_frame=results_to_plot[
        results_to_plot.variable.isin(["f1_score", "val_f1_score"])
    ],
    x="epochs",
    y="value",
    color="variable",
)

# %% [markdown] papermill={"duration": 1.975816, "end_time": "2022-03-05T13:47:02.322028", "exception": false, "start_time": "2022-03-05T13:47:00.346212", "status": "completed"} tags=[]
# ### Load best model, either phase 1 or 2

# %% papermill={"duration": 2.762513, "end_time": "2022-03-05T13:47:06.827574", "exception": false, "start_time": "2022-03-05T13:47:04.065061", "status": "completed"} tags=[]
best_phase2_f1 = results_tr["val_f1_score"].max()

if best_phase1_f1 > best_phase2_f1:
    effnet2_tfhub.load_weights(
        "training/" + "cp-" + f"{best_phase1_epoch}".rjust(4, "0") + ".ckpt"
    )
    print(f"best phase 1: {best_phase1_f1}")
else:
    print(f"best phase 2: {best_phase2_f1}")


# %% [markdown] papermill={"duration": 1.714644, "end_time": "2022-03-05T13:47:10.258351", "exception": false, "start_time": "2022-03-05T13:47:08.543707", "status": "completed"} tags=[]
# # IV. Submission
#
# id,label
# a762df180,0
# 24c5cf439,0
# 7581e896d,0
# eb4b03b29,0
# etc.

# %% papermill={"duration": 24.276779, "end_time": "2022-03-05T13:47:36.312376", "exception": false, "start_time": "2022-03-05T13:47:12.035597", "status": "completed"} tags=[]
test_pred = effnet2_tfhub.predict(ds_test, batch_size=batch_size)


# %% papermill={"duration": 42.696417, "end_time": "2022-03-05T13:48:20.729436", "exception": false, "start_time": "2022-03-05T13:47:38.033019", "status": "completed"} tags=[]
img_ids = []
img_preds = []
for imgs, idnum in ds_test:
    img_preds.append(effnet2_tfhub.predict(imgs, batch_size=batch_size))
    img_ids.append(idnum)

# %% papermill={"duration": 1.774333, "end_time": "2022-03-05T13:48:24.233518", "exception": false, "start_time": "2022-03-05T13:48:22.459185", "status": "completed"} tags=[]
img_ids = np.concatenate([img_id.numpy() for img_id in img_ids])


# %% papermill={"duration": 1.739725, "end_time": "2022-03-05T13:48:27.700953", "exception": false, "start_time": "2022-03-05T13:48:25.961228", "status": "completed"} tags=[]
img_preds = np.concatenate([img_pred.argmax(1) for img_pred in img_preds])

# %% papermill={"duration": 2.04976, "end_time": "2022-03-05T13:48:31.481003", "exception": false, "start_time": "2022-03-05T13:48:29.431243", "status": "completed"} tags=[]
img_ids.shape, img_preds.shape

# %% papermill={"duration": 1.745188, "end_time": "2022-03-05T13:48:35.002336", "exception": false, "start_time": "2022-03-05T13:48:33.257148", "status": "completed"} tags=[]
submission = pd.DataFrame({"id": img_ids, "label": img_preds})
submission["id"] = submission["id"].apply(lambda x: x.decode())

# %% papermill={"duration": 1.732708, "end_time": "2022-03-05T13:48:38.483096", "exception": false, "start_time": "2022-03-05T13:48:36.750388", "status": "completed"} tags=[]
submission.head()

# %% papermill={"duration": 2.011645, "end_time": "2022-03-05T13:48:42.226731", "exception": false, "start_time": "2022-03-05T13:48:40.215086", "status": "completed"} tags=[]
submission.dtypes

# %% papermill={"duration": 1.755977, "end_time": "2022-03-05T13:48:45.744302", "exception": false, "start_time": "2022-03-05T13:48:43.988325", "status": "completed"} tags=[]
submission.to_csv("submission.csv", index=False)

# %% papermill={"duration": 1.728528, "end_time": "2022-03-05T13:48:49.198230", "exception": false, "start_time": "2022-03-05T13:48:47.469702", "status": "completed"} tags=[]
