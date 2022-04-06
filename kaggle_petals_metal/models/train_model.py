import math
import logging
import tensorflow as tf
import click

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


import sys

sys.path.append("../../")


from kaggle_petals_metal.models.data_generator import DataGenerator
import tensorflow_hub as hub

# import plotly_express as px

print(tf.__version__)
print(tfa.__version__)


def get_effnet2_model(
    hyperparams={"lr": 0.001, "dropout": 0.2, "size": "small"}, image_size=224
):

    print(f"hyperparams: {hyperparams}")

    if hyperparams["size"] == "large":
        effnet2_base = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2"
    elif hyperparams["size"] == "medium":
        effnet2_base = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2"
    else:
        effnet2_base = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2"

    effnet2_tfhub = tf.keras.Sequential(
        [
            # Explicitly define the input shape so the model can be properly
            # loaded by the TFLiteConverter
            tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)),
            hub.KerasLayer(effnet2_base, trainable=False),
            tf.keras.layers.Dropout(rate=hyperparams["dropout"]),
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams["lr"]),
        loss="categorical_crossentropy",
        metrics=[
            tfa.metrics.F1Score(num_classes=104, average="macro"),
            tf.keras.metrics.CategoricalAccuracy(
                name="categorical_accuracy", dtype=None
            ),
        ],
    )

    return effnet2_tfhub


def get_model(
    model_type="effnet2",
    image_size=224,
    hyperparams={"lr": 0.001, "dropout": 0.2, "size": "small"},
):

    if model_type == "effnet2":
        return get_effnet2_model(hyperparams=hyperparams, image_size=image_size)


def train():

    image_size = 224
    batch_size = 64

    compute_steps_per_epoch = lambda x: int(math.ceil(1.0 * x / batch_size))
    steps_per_epoch_tr = compute_steps_per_epoch(12753)
    steps_per_epoch_val = compute_steps_per_epoch(3712)
    print(
        f"steps per train epoch: {steps_per_epoch_tr}, per val epoch {steps_per_epoch_val}"
    )

    tf.keras.backend.clear_session()

    ds_train, ds_valid, _ = DataGenerator(
        BATCH_SIZE=batch_size,
        IMAGE_SIZE=(image_size, image_size),
        RESIZE=None,
        tpu=False,
    ).get_datasets()

    model = get_model(
        model_type="effnet2",
        hyperparams={"lr": 0.001, "dropout": 0.2, size: "small"},
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

    history = model.fit(
        ds_train,
        epochs=2,
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


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main():

    train()


if __name__ == "__main__":

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
