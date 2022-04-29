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

# %% [markdown] papermill={"duration": 0.075647, "end_time": "2022-03-02T03:04:11.367231", "exception": false, "start_time": "2022-03-02T03:04:11.291584", "status": "completed"} tags=[]
# # EDA on Flower Classification with TPU Competition Dataset

# %% [markdown] papermill={"duration": 0.072906, "end_time": "2022-03-02T03:04:11.519765", "exception": false, "start_time": "2022-03-02T03:04:11.446859", "status": "completed"} tags=[]
# # I. Goals
#
# Classify images of flowers in 104 different classes. Classical image classification problem. Distinguish flowers which might be very similar in forms and colors. Images are from 5 public datasets
#
#
# * There appears to be a label hirarchy (flower type hirarchy. Some classes are very narrow, containing only a particular sub-type of flower (e.g. pink primroses) while other classes contain many sub-types (e.g. wild roses).
#
# * Metrics: Macro-F1 score does not take class-imbalance into account
# * Performance on public test set, there is no hidden set. Careful with overfitting

# %% [markdown] papermill={"duration": 0.070717, "end_time": "2022-03-02T03:04:11.665578", "exception": false, "start_time": "2022-03-02T03:04:11.594861", "status": "completed"} tags=[]
# # II. Data Extraction
#
# * Data is available local in Kaggle but also in a GC bucket. See below.
# * n TFRecord format.
#
# * same data in different resolution?
#
#

# %% papermill={"duration": 0.844068, "end_time": "2022-03-02T03:04:12.582215", "exception": false, "start_time": "2022-03-02T03:04:11.738147", "status": "completed"} tags=[]
# ! ls ../input/tpu-getting-started


# %% papermill={"duration": 6.85127, "end_time": "2022-03-02T03:04:19.506710", "exception": false, "start_time": "2022-03-02T03:04:12.655440", "status": "completed"} tags=[]
import tensorflow as tf

print(tf.__version__)
import pandas as pd
import seaborn as sns

# %% papermill={"duration": 0.645158, "end_time": "2022-03-02T03:04:20.223934", "exception": false, "start_time": "2022-03-02T03:04:19.578776", "status": "completed"} tags=[]
from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path("tpu-getting-started")
print(GCS_DS_PATH)  # what do gcs paths look like?

# %% [markdown] papermill={"duration": 0.070369, "end_time": "2022-03-02T03:04:20.366075", "exception": false, "start_time": "2022-03-02T03:04:20.295706", "status": "completed"} tags=[]
# names from https://www.kaggle.com/ryanholbrook/create-your-first-submission

# %% papermill={"duration": 0.090902, "end_time": "2022-03-02T03:04:20.527709", "exception": false, "start_time": "2022-03-02T03:04:20.436807", "status": "completed"} tags=[]
class_names = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "wild geranium",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",  # 00 - 09
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",  # 10 - 19
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",  # 20 - 29
    "carnation",
    "garden phlox",
    "love in the mist",
    "cosmos",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",  # 30 - 39
    "barberton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "daisy",
    "common dandelion",  # 40 - 49
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "lilac hibiscus",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",  # 50 - 59
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "iris",
    "windflower",
    "tree poppy",  # 60 - 69
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",  # 70 - 79
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen ",
    "watercress",
    "canna lily",  # 80 - 89
    "hippeastrum ",
    "bee balm",
    "pink quill",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",  # 90 - 99
    "trumpet creeper",
    "blackberry lily",
    "common tulip",
    "wild rose",
]  # 100 - 102
len(class_names)

# %% [markdown] papermill={"duration": 0.072, "end_time": "2022-03-02T03:04:20.671362", "exception": false, "start_time": "2022-03-02T03:04:20.599362", "status": "completed"} tags=[]
# # III. Meet & Greet Data
#
# For purpose of EDA focus partly on 512x512

# %% [markdown] papermill={"duration": 0.071394, "end_time": "2022-03-02T03:04:20.815195", "exception": false, "start_time": "2022-03-02T03:04:20.743801", "status": "completed"} tags=[]
# ## Categories of Flowers

# %% papermill={"duration": 0.084539, "end_time": "2022-03-02T03:04:20.971045", "exception": false, "start_time": "2022-03-02T03:04:20.886506", "status": "completed"} tags=[]
class_names

# %% [markdown] papermill={"duration": 0.072645, "end_time": "2022-03-02T03:04:21.117304", "exception": false, "start_time": "2022-03-02T03:04:21.044659", "status": "completed"} tags=[]
# Flowers exist in groups

# %% papermill={"duration": 0.083226, "end_time": "2022-03-02T03:04:21.278107", "exception": false, "start_time": "2022-03-02T03:04:21.194881", "status": "completed"} tags=[]
categories = ["lily", "rose", "iris", "tulip", "daisy", "poppy"]

# %% papermill={"duration": 0.08538, "end_time": "2022-03-02T03:04:21.438076", "exception": false, "start_time": "2022-03-02T03:04:21.352696", "status": "completed"} tags=[]
category_id_map = {name: i for i, name in enumerate(categories)}
id_count = max(category_id_map.values())
ids = []
for name in class_names:
    for cat in categories:
        if cat in name.split():
            ids.append(category_id_map[cat])
            break
    else:
        id_count += 1
        ids.append(id_count)

# %% papermill={"duration": 0.105937, "end_time": "2022-03-02T03:04:21.620373", "exception": false, "start_time": "2022-03-02T03:04:21.514436", "status": "completed"} tags=[]
class_groups = pd.DataFrame(zip(class_names, ids), columns=["names", "id"])
class_groups.groupby("id")["names"].apply(list).head(len(categories)).values

# %% [markdown] papermill={"duration": 0.075069, "end_time": "2022-03-02T03:04:21.771089", "exception": false, "start_time": "2022-03-02T03:04:21.696020", "status": "completed"} tags=[]
# > Some flowers are of the same type/category and hence expect classification errors among them.

# %% papermill={"duration": 0.083362, "end_time": "2022-03-02T03:04:21.929893", "exception": false, "start_time": "2022-03-02T03:04:21.846531", "status": "completed"} tags=[]
class_name_mapping = {i: name for i, name in enumerate(class_names)}


# %% [markdown] papermill={"duration": 0.07411, "end_time": "2022-03-02T03:04:22.079755", "exception": false, "start_time": "2022-03-02T03:04:22.005645", "status": "completed"} tags=[]
# ## Images

# %% papermill={"duration": 0.84108, "end_time": "2022-03-02T03:04:22.996950", "exception": false, "start_time": "2022-03-02T03:04:22.155870", "status": "completed"} tags=[]
# ! ls ../input/tpu-getting-started


# %% papermill={"duration": 0.123458, "end_time": "2022-03-02T03:04:23.195548", "exception": false, "start_time": "2022-03-02T03:04:23.072090", "status": "completed"} tags=[]
IMAGE_SIZE = [512, 512]

data_root = "../input/tpu-getting-started"
# data_root = GCS_DS_PATH

data_path = data_root + "/tfrecords-jpeg-512x512"


train_512 = tf.io.gfile.glob(data_path + "/train/*.tfrec")
val_512 = tf.io.gfile.glob(data_path + "/val/*.tfrec")
test_512 = tf.io.gfile.glob(data_path + "/test/*.tfrec")
all_512 = [train_512, val_512, test_512]

# %% [markdown] papermill={"duration": 0.074378, "end_time": "2022-03-02T03:04:23.345774", "exception": false, "start_time": "2022-03-02T03:04:23.271396", "status": "completed"} tags=[]
# 16 files per set

# %% papermill={"duration": 0.083159, "end_time": "2022-03-02T03:04:23.504563", "exception": false, "start_time": "2022-03-02T03:04:23.421404", "status": "completed"} tags=[]
[len(dset) for dset in all_512]

# %% papermill={"duration": 0.082614, "end_time": "2022-03-02T03:04:23.662175", "exception": false, "start_time": "2022-03-02T03:04:23.579561", "status": "completed"} tags=[]
train_512

# %% [markdown] papermill={"duration": 0.074677, "end_time": "2022-03-02T03:04:23.812004", "exception": false, "start_time": "2022-03-02T03:04:23.737327", "status": "completed"} tags=[]
# # IV. Univariate Analysis

# %% papermill={"duration": 0.086739, "end_time": "2022-03-02T03:04:23.977790", "exception": false, "start_time": "2022-03-02T03:04:23.891051", "status": "completed"} tags=[]
len(class_names)

# %% papermill={"duration": 0.109039, "end_time": "2022-03-02T03:04:24.177165", "exception": false, "start_time": "2022-03-02T03:04:24.068126", "status": "completed"} tags=[]
from tensorflow.data import Dataset, TFRecordDataset

# %% papermill={"duration": 0.186421, "end_time": "2022-03-02T03:04:24.448271", "exception": false, "start_time": "2022-03-02T03:04:24.261850", "status": "completed"} tags=[]
record_sample = TFRecordDataset(train_512)

# %% papermill={"duration": 41.172048, "end_time": "2022-03-02T03:05:05.696365", "exception": false, "start_time": "2022-03-02T03:04:24.524317", "status": "completed"} tags=[]
num_elements = 0
for element in record_sample:
    num_elements += 1
num_elements


# %% [markdown] papermill={"duration": 0.07651, "end_time": "2022-03-02T03:05:05.850403", "exception": false, "start_time": "2022-03-02T03:05:05.773893", "status": "completed"} tags=[]
# Image loading pipeline. References
#
# * https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
# * https://www.kaggle.com/ryanholbrook/create-your-first-submission

# %% papermill={"duration": 0.097464, "end_time": "2022-03-02T03:05:06.023906", "exception": false, "start_time": "2022-03-02T03:05:05.926442", "status": "completed"} tags=[]


def decode_image(image_data):
    # images are encoded as jpg
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = (
        tf.cast(image, tf.float32) / 255.0
    )  # convert image to floats in [0, 1] range
    # image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["class"], tf.int32)
    depth = tf.constant(104)
    # one_hot_encoded = tf.one_hot(indices=label, depth=depth)

    return image, label  # returns a dataset of (image, label) pairs


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    idnum = example["id"]
    return image, idnum  # returns a dataset of image(s)


def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = (
            True  # False # disable order, increase speed
        )

    AUTO = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=AUTO
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
    )
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


# %% papermill={"duration": 0.366346, "end_time": "2022-03-02T03:05:06.467604", "exception": false, "start_time": "2022-03-02T03:05:06.101258", "status": "completed"} tags=[]
ds_train_512 = load_dataset(train_512, labeled=True)
ds_val_512 = load_dataset(val_512, labeled=True)
ds_test_512 = load_dataset(test_512, labeled=False)


# %% papermill={"duration": 0.169568, "end_time": "2022-03-02T03:05:06.713787", "exception": false, "start_time": "2022-03-02T03:05:06.544219", "status": "completed"} tags=[]
for b, l in ds_train_512:
    break

# %% papermill={"duration": 0.088299, "end_time": "2022-03-02T03:05:06.878661", "exception": false, "start_time": "2022-03-02T03:05:06.790362", "status": "completed"} tags=[]
l


# %% papermill={"duration": 42.311114, "end_time": "2022-03-02T03:05:49.272119", "exception": false, "start_time": "2022-03-02T03:05:06.961005", "status": "completed"} tags=[]
def get_ds_size(dataset, dtype="train"):
    num_elements = 0
    labels = []
    for img, label in dataset:
        num_elements += 1
        labels.append(label.numpy())
    print(f"{dtype}: number of images: {num_elements}")
    if dtype != "test":
        return pd.Series([class_name_mapping[label] for label in labels])


ds_train_512_labels = (get_ds_size(ds_train_512, dtype="train"),)
ds_val_512_labels = get_ds_size(ds_val_512, dtype="val")
get_ds_size(ds_test_512, dtype="test")

# %% papermill={"duration": 0.113814, "end_time": "2022-03-02T03:05:49.476589", "exception": false, "start_time": "2022-03-02T03:05:49.362775", "status": "completed"} tags=[]
total = 12753 + 3712 + 7382
12753 / total, 3712 / total, 7382 / total


# %% [markdown] papermill={"duration": 0.092118, "end_time": "2022-03-02T03:05:49.662107", "exception": false, "start_time": "2022-03-02T03:05:49.569989", "status": "completed"} tags=[]
# > Test set is 2x the validation set in size

# %% [markdown] papermill={"duration": 0.093412, "end_time": "2022-03-02T03:05:49.847292", "exception": false, "start_time": "2022-03-02T03:05:49.753880", "status": "completed"} tags=[]
# ## Class Distribution

# %% papermill={"duration": 0.131535, "end_time": "2022-03-02T03:05:50.070865", "exception": false, "start_time": "2022-03-02T03:05:49.939330", "status": "completed"} tags=[]
def get_class_distr(ds_labels):
    ds_dist = pd.concat(
        [ds_labels.value_counts(), ds_labels.value_counts(normalize=True)], axis=1
    )
    ds_dist.columns = ["counts", "fraction"]
    return ds_dist


ds_train_512_labeldist = get_class_distr(ds_train_512_labels[0])
ds_val_512_labeldist = get_class_distr(ds_val_512_labels)

# %% papermill={"duration": 0.12667, "end_time": "2022-03-02T03:05:50.291656", "exception": false, "start_time": "2022-03-02T03:05:50.164986", "status": "completed"} tags=[]
ds_train_512_labeldist

# %% papermill={"duration": 0.108127, "end_time": "2022-03-02T03:05:50.489958", "exception": false, "start_time": "2022-03-02T03:05:50.381831", "status": "completed"} tags=[]
ds_train_512_labeldist.head(20)

# %% [markdown] papermill={"duration": 0.085708, "end_time": "2022-03-02T03:05:50.676132", "exception": false, "start_time": "2022-03-02T03:05:50.590424", "status": "completed"} tags=[]
# Problem Classes: 27 classes with less than 10 images in the training set!

# %% papermill={"duration": 0.101933, "end_time": "2022-03-02T03:05:50.860040", "exception": false, "start_time": "2022-03-02T03:05:50.758107", "status": "completed"} tags=[]
ds_val_512_labeldist.tail(28)

# %% papermill={"duration": 0.09518, "end_time": "2022-03-02T03:05:51.035114", "exception": false, "start_time": "2022-03-02T03:05:50.939934", "status": "completed"} tags=[]
problem_classes = ds_val_512_labeldist.tail(27).index
problem_classes

# %% papermill={"duration": 0.128819, "end_time": "2022-03-02T03:05:51.288316", "exception": false, "start_time": "2022-03-02T03:05:51.159497", "status": "completed"} tags=[]

# %% papermill={"duration": 0.131957, "end_time": "2022-03-02T03:05:51.546422", "exception": false, "start_time": "2022-03-02T03:05:51.414465", "status": "completed"} tags=[]
import numpy as np

# %% papermill={"duration": 0.182388, "end_time": "2022-03-02T03:05:51.854589", "exception": false, "start_time": "2022-03-02T03:05:51.672201", "status": "completed"} tags=[]
for key in ds_val_512_labeldist.to_dict()["fraction"].keys():
    if not np.isclose(
        ds_val_512_labeldist.to_dict()["fraction"][key],
        ds_val_512_labeldist.to_dict()["fraction"][key],
    ):
        print(f"{key} not close")


# %% [markdown] papermill={"duration": 0.127662, "end_time": "2022-03-02T03:05:52.107563", "exception": false, "start_time": "2022-03-02T03:05:51.979901", "status": "completed"} tags=[]
# > * Classes are highly imbalanced, in fact some have only 18 images in train, and 5 images in valid set!
# > * Majority class makes up only 6% of the whole data.
# > * class distribution in train and valid is the same, as it should be.

# %% papermill={"duration": 0.495038, "end_time": "2022-03-02T03:05:52.729007", "exception": false, "start_time": "2022-03-02T03:05:52.233969", "status": "completed"} tags=[]
ds_train_512_labeldist["counts"].plot(kind="hist")

# %% papermill={"duration": 0.22368, "end_time": "2022-03-02T03:05:53.039408", "exception": false, "start_time": "2022-03-02T03:05:52.815728", "status": "completed"} tags=[]
sns.boxplot(x=ds_train_512_labeldist["counts"])

# %% papermill={"duration": 0.098768, "end_time": "2022-03-02T03:05:53.235976", "exception": false, "start_time": "2022-03-02T03:05:53.137208", "status": "completed"} tags=[]
ds_train_512_labeldist["counts"].median()

# %% [markdown] papermill={"duration": 0.08985, "end_time": "2022-03-02T03:05:53.414261", "exception": false, "start_time": "2022-03-02T03:05:53.324411", "status": "completed"} tags=[]
# > 9 classes have large number of images (outliers above, above ~280) while the median is 88 images per class

# %% [markdown] papermill={"duration": 0.087999, "end_time": "2022-03-02T03:05:53.593221", "exception": false, "start_time": "2022-03-02T03:05:53.505222", "status": "completed"} tags=[]
# ## Images Visual Analysis

# %% papermill={"duration": 0.11084, "end_time": "2022-03-02T03:05:53.793953", "exception": false, "start_time": "2022-03-02T03:05:53.683113", "status": "completed"} tags=[]
from matplotlib import pyplot as plt
import math


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
            class_names[label],
            "OK" if correct else "NO",
            "\u2192" if not correct else "",
            class_names[correct_label] if not correct else "",
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


def display_batch_of_images(databatch, predictions=None, FIGSIZE=13):
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
        title = "" if label is None else class_names[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
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


# %% papermill={"duration": 0.102433, "end_time": "2022-03-02T03:05:53.984018", "exception": false, "start_time": "2022-03-02T03:05:53.881585", "status": "completed"} tags=[]
ds_train_512

# %% papermill={"duration": 0.090814, "end_time": "2022-03-02T03:05:54.162920", "exception": false, "start_time": "2022-03-02T03:05:54.072106", "status": "completed"} tags=[]


# %% [markdown] papermill={"duration": 0.089524, "end_time": "2022-03-02T03:05:54.341016", "exception": false, "start_time": "2022-03-02T03:05:54.251492", "status": "completed"} tags=[]
# ### Random sample of flowers

# %% papermill={"duration": 1.857271, "end_time": "2022-03-02T03:05:56.290105", "exception": false, "start_time": "2022-03-02T03:05:54.432834", "status": "completed"} tags=[]
ds_train_512 = load_dataset(train_512, labeled=True)

ds_train_512 = ds_train_512.batch(10)
bs = next(iter(ds_train_512))

display_batch_of_images(bs)

# %% [markdown] papermill={"duration": 0.115404, "end_time": "2022-03-02T03:05:56.522913", "exception": false, "start_time": "2022-03-02T03:05:56.407509", "status": "completed"} tags=[]
# Findings
#
# * details of background visible
# * some images have slighly blurry flowers
# * Images appear to stem from the outside, garden, nature etc.
# * sometime its one flower, sometimes multiple
# * picture angle on plant(s) seem fairly arbitray
#

# %% papermill={"duration": 5.35614, "end_time": "2022-03-02T03:06:01.997862", "exception": false, "start_time": "2022-03-02T03:05:56.641722", "status": "completed"} tags=[]
ds_train_512 = load_dataset(train_512, labeled=True)

ds_train_512 = ds_train_512.batch(40)
bs = next(iter(ds_train_512))

display_batch_of_images(bs)

# %% [markdown] papermill={"duration": 0.175346, "end_time": "2022-03-02T03:06:02.337977", "exception": false, "start_time": "2022-03-02T03:06:02.162631", "status": "completed"} tags=[]
# Analysis:
# * Indoor plants are also possible!
# * close up shots showing only part of the flower exist
# * Not clear if flowers are in different blooming stages
# * insects on flowers
# * flowers within category can differ signiifcantly: geranium vs wild geranium

# %% [markdown] papermill={"duration": 0.180527, "end_time": "2022-03-02T03:06:02.714235", "exception": false, "start_time": "2022-03-02T03:06:02.533708", "status": "completed"} tags=[]
# ## Flowers by Class

# %% papermill={"duration": 0.162929, "end_time": "2022-03-02T03:06:03.040166", "exception": false, "start_time": "2022-03-02T03:06:02.877237", "status": "completed"} tags=[]
# class_name_mapping

# %% papermill={"duration": 0.163126, "end_time": "2022-03-02T03:06:03.360910", "exception": false, "start_time": "2022-03-02T03:06:03.197784", "status": "completed"} tags=[]
inverse_class_name_mapping = {class_name_mapping[i]: i for i in class_name_mapping}

# %% papermill={"duration": 0.167966, "end_time": "2022-03-02T03:06:03.684974", "exception": false, "start_time": "2022-03-02T03:06:03.517008", "status": "completed"} tags=[]
from tqdm import tqdm
from numpy.random import default_rng


# %% papermill={"duration": 0.167694, "end_time": "2022-03-02T03:06:04.021302", "exception": false, "start_time": "2022-03-02T03:06:03.853608", "status": "completed"} tags=[]

# %% papermill={"duration": 0.182481, "end_time": "2022-03-02T03:06:04.369035", "exception": false, "start_time": "2022-03-02T03:06:04.186554", "status": "completed"} tags=[]
def display_batch_by_class(files, name="iris", top_n=10, FIGSIZE=13):

    class_idx = inverse_class_name_mapping[name]
    print(class_idx)

    max_imgs_per_class = ds_val_512_labeldist.loc[name, "counts"]

    if top_n > max_imgs_per_class:
        top_n = max_imgs_per_class
        print(
            f"warning, class has only {max_imgs_per_class} images. Show all images for class"
        )

    # get position of class images in dataset
    sample_idx = []

    ds = load_dataset(files, labeled=True)
    ds = ds.batch(1)
    for i, (img, label) in tqdm(enumerate(ds)):
        if label.numpy()[0] == class_idx:
            sample_idx.append(i)

    # choose randomly top_n images
    rng = default_rng(42)
    sample_idx_shuffled = sample_idx.copy()
    rng.shuffle(sample_idx_shuffled)
    top_n_sample = sample_idx_shuffled[:top_n]

    ds = load_dataset(files, labeled=True)
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


# %% [markdown] papermill={"duration": 0.159337, "end_time": "2022-03-02T03:06:04.686476", "exception": false, "start_time": "2022-03-02T03:06:04.527139", "status": "completed"} tags=[]
# #### Most Common Class: Iris

# %% papermill={"duration": 47.74569, "end_time": "2022-03-02T03:06:52.597712", "exception": false, "start_time": "2022-03-02T03:06:04.852022", "status": "completed"} tags=[]
display_batch_by_class(train_512, name="iris", top_n=10)

# %% [markdown] papermill={"duration": 0.365996, "end_time": "2022-03-02T03:06:53.324589", "exception": false, "start_time": "2022-03-02T03:06:52.958593", "status": "completed"} tags=[]
# ### Problem Classes

# %% [markdown] papermill={"duration": 0.396846, "end_time": "2022-03-02T03:06:54.089168", "exception": false, "start_time": "2022-03-02T03:06:53.692322", "status": "completed"} tags=[]
# #### Siam Tulip - one of the least common classes with only 5 images

# %% papermill={"duration": 48.046829, "end_time": "2022-03-02T03:07:42.503814", "exception": false, "start_time": "2022-03-02T03:06:54.456985", "status": "completed"} tags=[]
display_batch_by_class(train_512, name="siam tulip", top_n=20)

# %% [markdown] papermill={"duration": 0.561562, "end_time": "2022-03-02T03:07:43.614564", "exception": false, "start_time": "2022-03-02T03:07:43.053002", "status": "completed"} tags=[]
# * **Danger**: Is there something common in their background which could mislead the algorithm to use wrong features for identification? This class is especially prone due to the low number of images

# %% papermill={"duration": 49.026561, "end_time": "2022-03-02T03:08:33.203483", "exception": false, "start_time": "2022-03-02T03:07:44.176922", "status": "completed"} tags=[]
display_batch_by_class(train_512, name="moon orchid", top_n=20)

# %% [markdown] papermill={"duration": 0.750047, "end_time": "2022-03-02T03:08:34.700949", "exception": false, "start_time": "2022-03-02T03:08:33.950902", "status": "completed"} tags=[]
# ### Look at all problem classes

# %% papermill={"duration": 0.763372, "end_time": "2022-03-02T03:08:36.221704", "exception": false, "start_time": "2022-03-02T03:08:35.458332", "status": "completed"} tags=[]
problem_classes

# %% papermill={"duration": 1320.026565, "end_time": "2022-03-02T03:30:37.008866", "exception": false, "start_time": "2022-03-02T03:08:36.982301", "status": "completed"} tags=[]
for class_name in problem_classes:
    display_batch_by_class(train_512, name=class_name, top_n=10, FIGSIZE=6)

# %% [markdown] papermill={"duration": 5.682863, "end_time": "2022-03-02T03:30:48.539928", "exception": false, "start_time": "2022-03-02T03:30:42.857065", "status": "completed"} tags=[]
# Analysis for problem classes:
# * same shot, from front, one plant only: hard-leaved pocket
# * there can be still large variation in color and shape for each image per class.
# * some classes have few images with similar background

# %% papermill={"duration": 5.784354, "end_time": "2022-03-02T03:31:00.040664", "exception": false, "start_time": "2022-03-02T03:30:54.256310", "status": "completed"} tags=[]

# %% papermill={"duration": 5.78054, "end_time": "2022-03-02T03:31:11.432100", "exception": false, "start_time": "2022-03-02T03:31:05.651560", "status": "completed"} tags=[]

# %% [markdown] papermill={"duration": 5.717909, "end_time": "2022-03-02T03:31:22.901572", "exception": false, "start_time": "2022-03-02T03:31:17.183663", "status": "completed"} tags=[]
# ## Impact of Resolution
#
# * How does the image attributes change when decreasing the resolution?
# * Which features are not visible anymore?

# %% papermill={"duration": 5.846494, "end_time": "2022-03-02T03:31:34.393218", "exception": false, "start_time": "2022-03-02T03:31:28.546724", "status": "completed"} tags=[]
files_all_res = [
    tf.io.gfile.glob(data_root + "/tfrecords-jpeg-512x512" + "/train/*.tfrec"),
    tf.io.gfile.glob(data_root + "/tfrecords-jpeg-331x331" + "/train/*.tfrec"),
    tf.io.gfile.glob(data_root + "/tfrecords-jpeg-224x224" + "/train/*.tfrec"),
    tf.io.gfile.glob(data_root + "/tfrecords-jpeg-192x192" + "/train/*.tfrec"),
]
resolutions = [512, 331, 224, 192]


# %% [markdown] papermill={"duration": 5.718232, "end_time": "2022-03-02T03:31:45.769071", "exception": false, "start_time": "2022-03-02T03:31:40.050839", "status": "completed"} tags=[]
# Compare impact of resolution on images by comparing the same image of flowers.
# Unfortunately the images are not in the same order for different resolutions
# and no unique flower id exists to link the images of different resolution.
# Hence I pick one class for which I can plot all flower images.

# %% papermill={"duration": 129.031434, "end_time": "2022-03-02T03:34:00.437712", "exception": false, "start_time": "2022-03-02T03:31:51.406278", "status": "completed"} tags=[]
for res, files in zip(resolutions, files_all_res):
    # ds = load_dataset(files, labeled=True)
    print(f"Image Resolution: {res}")
    display_batch_by_class(files, name="moon orchid", top_n=20)
    # ds = ds.batch(1)
    # batch = next(iter(ds))
    # print(res)
    # display_batch_of_images(batch)


# %% [markdown] papermill={"duration": 6.182112, "end_time": "2022-03-02T03:34:12.877765", "exception": false, "start_time": "2022-03-02T03:34:06.695653", "status": "completed"} tags=[]
# ### Augmentation Strategy
#
# * Images appear blurry: blurr
# * Images are zoomed in and out: zoom in/out
# * Images are brighter and darker

# %% papermill={"duration": 6.217261, "end_time": "2022-03-02T03:34:25.196960", "exception": false, "start_time": "2022-03-02T03:34:18.979699", "status": "completed"} tags=[]
