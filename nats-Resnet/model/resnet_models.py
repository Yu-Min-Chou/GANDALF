from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
import numpy as np
import math
import config

import tensorflow as tf
import tensorflow_addons as tfa
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

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride,
                                                       use_bias=False))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block

class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=config.n_class, activation=tf.keras.activations.softmax, use_bias=False)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        x = self.flatten(x)
        output = self.fc(x)

        return output

def resnet_18():
    return ResNetTypeI(layer_params=[2, 2, 2, 2])

class Resnet:
    def __init__(self):
        self.epochs = config.local_epochs
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.channel_size = config.channel_size
        self.C = resnet_18()
        self.C.build(input_shape=(None, self.image_size, self.image_size, self.channel_size))
        self.weight = None
        # self.C.summary()


    def serialize(self):
        cnn_bytes = {"weights": self.weight}
        outputfile = 'well-trained_{}'.format(config.dataset)
        fw = open(outputfile, 'wb')
        pickle.dump(cnn_bytes, fw)
        fw.close()

    def get_model_weight(self):
        return self.C.get_weights()

    def set_model_weight(self, C_weight):
        self.C.set_weights(C_weight)

    def test(self, dataset_test):
        sparse_categorical_accuracy = metrics.SparseCategoricalAccuracy()
        sparse_categorical_accuracy.reset_states()
        c_accu_result = 0

        for x_test, y_test in dataset_test:
            logits = self.C(x_test)
            sparse_categorical_accuracy.update_state(y_true=y_test, y_pred=logits)

        c_accu_result = sparse_categorical_accuracy.result() * 100
        

        return c_accu_result.numpy()

    def predict(self, samples):
        return self.C(samples, training=False)
    
    def train(self, dataset_train):
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum = 0.9, decay=1e-4)
        train_sparse_categorical_accuracy = metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_sparse_categorical_accuracy = metrics.SparseCategoricalAccuracy(name='test_accuracy')

        c_loss_result = 0
        c_accu_result = 0

        for epoch in range(self.epochs):

            print("\nEpoch: {}/{}\n".format(epoch, self.epochs))
            num_batch = 0
            train_sparse_categorical_accuracy.reset_states()

            for x, y in dataset_train:
                num_batch = num_batch + 1
                with tf.GradientTape() as t:
                    logits = self.C(x, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=logits)
                    loss = tf.reduce_mean(loss)

                grad = t.gradient(loss, self.C.trainable_variables)
                optimizer.apply_gradients(zip(grad, self.C.trainable_variables))
                train_sparse_categorical_accuracy.update_state(y_true=y, y_pred=logits)

                if num_batch % 10 == 0:
                    percentage = int(round(num_batch/len(dataset_train)*100))
                    bar_len = 29
                    filled_len = int((bar_len*percentage)/100)
                    bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)
                    msg = "Step: {:>5} - [{}]  {:>3}% - acc: {:.4f} - loss: {:.4f}"
                    print(msg.format(num_batch, bar, percentage, train_sparse_categorical_accuracy.result(), loss))
                    train_sparse_categorical_accuracy.reset_states()

            print("#################################################################################")
        
        test_sparse_categorical_accuracy.reset_states()
        for x_test, y_test in dataset_train:
            logits = self.C(x_test)
            test_sparse_categorical_accuracy.update_state(y_true=y_test, y_pred=logits)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=logits)
            c_loss_result += tf.reduce_mean(loss)
        
        c_loss_result /= len(dataset_train)
        c_accu_result = test_sparse_categorical_accuracy.result() * 100

        return c_loss_result.numpy(), c_accu_result.numpy()