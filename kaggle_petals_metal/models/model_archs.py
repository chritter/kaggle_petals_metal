"""contains model architectures"""

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

print(tf.__version__)
print(tfa.__version__)


def get_effnet2_model(
    hyperparams={"lr": 0.001, "dropout": 0.2, "size": "small", "label_smoothing": 0.0},
    image_size=224,
):

    print("load effnetv2 model")
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

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, label_smoothing=hyperparams["label_smoothing"]
    )

    effnet2_tfhub.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams["lr"]),
        loss=loss,
        metrics=[
            tfa.metrics.F1Score(num_classes=104, average="macro"),
            tf.keras.metrics.CategoricalAccuracy(
                name="categorical_accuracy", dtype=None
            ),
        ],
    )

    return effnet2_tfhub


def get_vits_16_model(
    hyperparams={"lr": 0.001, "dropout": 0.2, "size": "small", "label_smoothing": 0.0},
    image_size=224,
):

    print("load vits 16 model")
    print(f"hyperparams: {hyperparams}")

    vits_base = "https://tfhub.dev/sayakpaul/vit_s16_fe/1"
    vits_base = "https://tfhub.dev/sayakpaul/vit_b8_fe/1"

    vits_16_tfhub = tf.keras.Sequential(
        [
            # Explicitly define the input shape so the model can be properly
            # loaded by the TFLiteConverter
            tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)),
            hub.KerasLayer(vits_base, trainable=False),
            tf.keras.layers.Dropout(rate=hyperparams["dropout"]),
            tf.keras.layers.Dense(104, activation="softmax"),
        ]
    )
    vits_16_tfhub.build(
        (
            None,
            image_size,
            image_size,
            3,
        )
    )  # This is to be used for subclassed models, which do not know at instantiation time what their inputs look like.

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, label_smoothing=hyperparams["label_smoothing"]
    )

    vits_16_tfhub.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams["lr"]),
        loss=loss,
        metrics=[
            tfa.metrics.F1Score(num_classes=104, average="macro"),
            tf.keras.metrics.CategoricalAccuracy(
                name="categorical_accuracy", dtype=None
            ),
        ],
    )

    return vits_16_tfhub
