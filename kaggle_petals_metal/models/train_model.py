import logging
import math
import sys

import click
import tensorflow as tf

from kaggle_petals_metal.models.data_generator import DataGenerator
from kaggle_petals_metal.models.model_archs import (
    get_effnet2_model,
    get_vits_16_model,
)

# sys.path.append("../../")


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


def get_model(
    model_arch="effnet2",
    image_size=224,
    hyperparams={"lr": 0.001, "dropout": 0.2, "size": "small", "label_smoothing": 0.0},
):

    if model_arch == "effnet2":
        return get_effnet2_model(hyperparams=hyperparams, image_size=image_size)
    elif model_arch == "vits_16":
        raise NotImplementedError("Vits 16 model does not work on Mac M1")
        # return get_vits_16_model(hyperparams=hyperparams, image_size=image_size)


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
        hyperparams={"lr": 0.001, "dropout": 0.2, "size": "small"},
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
