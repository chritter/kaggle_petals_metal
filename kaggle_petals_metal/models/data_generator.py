"""data generator functions for deep learining models"""

from functools import partial
import pathlib

import tensorflow as tf

from kaggle_petals_metal.data.get_class_names import get_class_names

print("Data generator uses Tensorflow version " + tf.__version__)


CLASSES = get_class_names()
AUTO = tf.data.experimental.AUTOTUNE
CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()


class DataGenerator:
    def __init__(self, BATCH_SIZE, IMAGE_SIZE, RESIZE, with_id=False, tpu=False):

        self.BATCH_SIZE = BATCH_SIZE
        self.IMAGE_SIZE = IMAGE_SIZE
        self.RESIZE = RESIZE
        self.with_id = with_id
        self.tpu = tpu

    def decode_image(self, image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = (
            tf.cast(image, tf.float32) / 255.0
        )  # convert image to floats in [0, 1] range
        image = tf.reshape(image, [*self.IMAGE_SIZE, 3])  # explicit size needed for TPU

        if self.RESIZE:
            target_height, target_width = self.RESIZE
            image = tf.image.resize_with_pad(
                image,
                target_height,
                target_width  # method=ResizeMethod.BILINEAR,
                # antialias=False
            )

        return image

    def read_labeled_tfrecord(self, example):
        LABELED_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
            "class": tf.io.FixedLenFeature(
                [], tf.int64
            ),  # shape [] means single element
            "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        }
        example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
        image = self.decode_image(example["image"])
        label = tf.cast(example["class"], tf.int32)
        depth = tf.constant(104)

        one_hot_encoded = tf.one_hot(indices=label, depth=depth)

        if not self.with_id:
            return image, one_hot_encoded  # returns a dataset of (image, label) pairs
        else:
            image_id = tf.cast(example["id"], tf.string)
            return image, one_hot_encoded, image_id

    def read_unlabeled_tfrecord(self, example):
        UNLABELED_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
            "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
            # class is missing, this competitions's challenge is to predict flower classes for the test dataset
        }
        example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
        image = self.decode_image(example["image"])
        idnum = example["id"]
        return image, idnum  # returns a dataset of image(s)

    def load_dataset(self, filenames, labeled=True, ordered=False):
        # Read from TFRecords. For optimal performance, reading from multiple files at once and
        # disregarding data order. Order does not matter since we will be shuffling the data anyway.

        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = (
                False  # disable order, increase speed
            )

        dataset = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=1
        )  # AUTO) # automatically interleaves reads from multiple files
        dataset = dataset.with_options(
            ignore_order
        )  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(
            self.read_labeled_tfrecord if labeled else self.read_unlabeled_tfrecord,
            num_parallel_calls=1,
        )  # AUTO)
        # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
        return dataset

    def data_augment(self, image, label):
        # Thanks to the dataset.prefetch(AUTO)
        # statement in the next function (below), this happens essentially
        # for free on TPU. Data pipeline code is executed on the "CPU"
        # part of the TPU while the TPU itself is computing gradients.
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_saturation(image, 0, 2)
        return image, label

    def get_training_dataset(self, TRAINING_FILENAMES):
        dataset = self.load_dataset(TRAINING_FILENAMES, labeled=True)
        # dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        dataset = (
            dataset.repeat()
        )  # the training dataset must repeat for several epochs
        dataset = dataset.shuffle(2048)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(
            AUTO
        )  # prefetch next batch while training (autotune prefetch buffer size)
        return dataset

    def get_validation_dataset(self, VALIDATION_FILENAMES, ordered=False):
        dataset = self.load_dataset(
            VALIDATION_FILENAMES,
            labeled=True,
            ordered=ordered,
        )
        dataset = dataset.batch(self.BATCH_SIZE)
        # dataset = dataset.cache()
        # dataset = dataset.prefetch(AUTO)
        return dataset

    def get_test_dataset(self, TEST_FILENAMES, ordered=False):
        dataset = self.load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(AUTO)
        return dataset

    def get_datasets(self):

        if not self.tpu:
            data_root = CURRENT_FILE_PATH / "../../data/raw"
        else:
            # data_root = GCS_DS_PATH
            print("tpu not implemented yet")
            return

        if 512 in self.IMAGE_SIZE:

            data_path = data_root / "tfrecords-jpeg-512x512"
            # IMAGE_SIZE = [512, 512]
        elif 224 in self.IMAGE_SIZE:
            data_path = data_root / "tfrecords-jpeg-224x224"
            # IMAGE_SIZE = [224, 224]
        elif 331 in self.IMAGE_SIZE:
            data_path = data_root / "tfrecords-jpeg-331x331"
            # IMAGE_SIZE = [331, 331]
        elif 192 in self.IMAGE_SIZE:
            data_path = data_root / "tfrecords-jpeg-192x192"
            # IMAGE_SIZE = [192, 192]
        else:
            print("wrong image size")
            return

        TRAINING_FILENAMES = tf.io.gfile.glob(str(data_path / "train/*.tfrec"))
        VALIDATION_FILENAMES = tf.io.gfile.glob(str(data_path / "val/*.tfrec"))
        TEST_FILENAMES = tf.io.gfile.glob(str(data_path / "test/*.tfrec"))

        ds_train = self.get_training_dataset(TRAINING_FILENAMES)
        ds_valid = self.get_validation_dataset(VALIDATION_FILENAMES)
        ds_test = self.get_test_dataset(TEST_FILENAMES)

        print("Training:", ds_train)
        print("Validation:", ds_valid)
        print("Test:", ds_test)

        return ds_train, ds_valid, ds_test
