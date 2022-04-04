"""testing data generator"""

from unittest import TestCase
from unittest import mock
import tensorflow as tf

from kaggle_petals_metal.models.data_generator import DataGenerator


class test_datagenerator(TestCase):
    def test_datagenerator_get_datasets_standard(self):

        batch_size = 2
        image_size = 224

        data_gen = DataGenerator(
            BATCH_SIZE=batch_size,
            IMAGE_SIZE=(image_size, image_size),
            RESIZE=None,
            tpu=False,
            with_id=False,
        )
        ds_train, ds_valid, ds_test = data_gen.get_datasets()

        self.assertEqual(
            ds_train.element_spec,
            (
                tf.TensorSpec(
                    shape=(None, image_size, image_size, 3), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(None, 104), dtype=tf.float32),
            ),
        )

        self.assertEqual(
            ds_valid.element_spec,
            (
                tf.TensorSpec(
                    shape=(None, image_size, image_size, 3), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(None, 104), dtype=tf.float32),
            ),
        )

        self.assertEqual(
            ds_test.element_spec,
            (
                tf.TensorSpec(
                    shape=(None, image_size, image_size, 3), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(None,), dtype=tf.string),
            ),
        )
