"""functions for visualization of images"""

import math
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from numpy.random import default_rng

import sys

sys.path.append("../../")

from kaggle_petals_metal.data.get_class_names import get_class_names
from kaggle_petals_metal.models.data_generator import DataGenerator


CLASSES = get_class_names()


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec
    # files, i.e. flowers00-230.tfrec = 230 data items
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)


def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object:  # binary string in this case,
        # these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is
    # the case for test data)
    return numpy_images, numpy_labels


def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return CLASSES[label], True
    correct = label == correct_label
    return (
        "{} [{}{}{}]".format(
            CLASSES[label],
            "OK" if correct else "NO",
            "\u2192" if not correct else "",
            CLASSES[correct_label] if not correct else "",
        ),
        correct,
    )


def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis("off")
    plt.imshow(image)
    if len(title) > 0:
        plt.title(
            title,
            fontsize=int(titlesize) if not red else int(titlesize / 1.2),
            color="red" if red else "black",
            fontdict={"verticalalignment": "center"},
            pad=int(titlesize / 1.5),
        )
    return (subplot[0], subplot[1], subplot[2] + 1)


def display_training_curves(training, validation, title, subplot):
    if subplot % 10 == 1:  # set up the subplots on the first call
        plt.subplots(figsize=(10, 10), facecolor="#F0F0F0")
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor("#F8F8F8")
    ax.plot(training)
    ax.plot(validation)
    ax.set_title("model " + title)
    ax.set_ylabel(title)
    # ax.set_ylim(0.28,1.05)
    ax.set_xlabel("epoch")
    ax.legend(["train", "valid."])


def display_batch_of_images(databatch, predictions=None, FIGSIZE=13, image_ids=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]

    # auto-squaring: this will drop data that does not fit into square
    # or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows + 1

    # size and spacing
    # FIGSIZE = 13.0
    SPACING = 0.1
    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

    # display
    for i, (image, label) in enumerate(
        zip(images[: rows * cols], labels[: rows * cols])
    ):
        title = "" if label is None else CLASSES[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        if image_ids is not None:
            title = title + "\n" + image_ids[i]
        dynamic_titlesize = (
            FIGSIZE * SPACING / max(rows, cols) * 40 + 3
        )  # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(
            image, title, subplot, not correct, titlesize=dynamic_titlesize
        )

    # layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()


def display_batch_by_class(files, name="iris", top_n=10, FIGSIZE=13):

    class_name_mapping = {i: name for i, name in enumerate(CLASSES)}
    inverse_class_name_mapping = {class_name_mapping[i]: i for i in class_name_mapping}
    class_idx = inverse_class_name_mapping[name]
    print(class_idx)

    # get position of class images in dataset
    sample_idx = []

    ds = DataGenerator(BATCH_SIZE=32, IMAGE_SIZE=(224, 224), RESIZE=None).load_dataset(
        files, labeled=True
    )
    ds = ds.batch(1)
    for i, (img, label) in tqdm(enumerate(ds)):
        # print(label)
        if label.numpy()[0].argmax() == class_idx:
            sample_idx.append(i)

    print(f"found {len(sample_idx)} images and take sample of {top_n} images")
    # choose randomly top_n images
    rng = default_rng(42)
    sample_idx_shuffled = sample_idx.copy()
    rng.shuffle(sample_idx_shuffled)
    top_n_sample = sample_idx_shuffled[:top_n]

    ds = DataGenerator(BATCH_SIZE=32, IMAGE_SIZE=(224, 224), RESIZE=None).load_dataset(
        files, labeled=True
    )
    ds = ds.batch(1)
    # get thte images for each data point
    images_class = []
    tmp = []
    for i, (img, label) in tqdm(enumerate(ds)):
        if i in top_n_sample:
            images_class.append(img)
            tmp.append(label)

    batch = tf.stack([tf.squeeze(img) for img in images_class]), tf.stack(
        [class_idx for i in range(len(images_class))]
    )

    display_batch_of_images(batch, FIGSIZE=FIGSIZE)
