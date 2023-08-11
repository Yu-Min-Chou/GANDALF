from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
import numpy as np
import math
import config

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import random
from tensorflow.python.keras import layers, Sequential
from tensorflow.python.keras import metrics
from tensorflow.python.keras import models

from tensorflow.python.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D,BatchNormalization
from tensorflow.python.keras.layers import Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow import optimizers

AUTOTUNE = tf.data.experimental.AUTOTUNE

class DatasetPipeline:
    def __init__(self):
        self.dataset_name = config.dataset
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.channel_size = config.channel_size
        self.dataset_info = {}
        self.data_augmentation = tf.keras.Sequential([
                layers.ZeroPadding2D(padding=2),
                layers.RandomCrop(self.image_size, self.image_size),
                layers.RandomRotation(0.1)
            ])
        self.data_augmentation.build(input_shape=(None, self.image_size, self.image_size, self.channel_size))

    def preprocess_image(self, image, label):
        image = tf.image.resize(image, (self.image_size, self.image_size))
        image = (tf.dtypes.cast(image, tf.float32) / 127.5) - 1.0
        label = tf.dtypes.cast(label, tf.int32)

        return image, label

    def dataset_cache(self, dataset):
        tmp_dir = Path(tempfile.gettempdir())
        cache_dir = tmp_dir.joinpath('cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        for p in cache_dir.glob(self.dataset_name + '*'):
            p.unlink()
        return dataset.cache(str(cache_dir / self.dataset_name))

    def load_dataset(self):
        if(self.dataset_name == "emnist"):
            ds, self.dataset_info = tfds.load(name='emnist/balanced',
                                            with_info=True,
                                            as_supervised=True)
            ds_train, ds_test = ds["train"], ds["test"]
            ds_train = ds_train.map(lambda x, y: self.preprocess_image(x,y), AUTOTUNE)
            ds_test = ds_test.map(lambda x, y: self.preprocess_image(x, y), AUTOTUNE)
        else if(self.dataset_name == "cinic10"):
            shape_x = (config.image_size, config.image_size, config.channel_size)
            shape_y = ()
            spec = (tf.TensorSpec(shape_x, dtype = tf.float32), tf.TensorSpec(shape_y, dtype = tf.int32))
            path = '/cinic10/train'
            ds_train = tf.data.experimental.load(path, spec)
            path = '/cinic10/test'
            ds_test = tf.data.experimental.load(path, spec)
        ds_train = ds_train.shuffle(50000, reshuffle_each_iteration=True)
        ds_train = ds_train.batch(self.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
        ds_train = ds_train.map(lambda x, y: (self.data_augmentation(x, training=True), y))
        ds_test = ds_test.shuffle(50000, reshuffle_each_iteration=True)
        ds_test = ds_test.batch(self.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
        
        return ds_train, ds_test