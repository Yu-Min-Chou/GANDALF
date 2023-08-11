from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import pickle
import tempfile
import math
from functools import partial
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import random
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import models

from model import ops
from model.utils import img_merge
from model.utils import pbar
from model.utils import save_image_grid

import config

AUTOTUNE = tf.data.experimental.AUTOTUNE


class WGANGP:
    def __init__(self, total_images):
        self.z_dim = config.z_size
        self.epochs = config.local_epochs
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.n_critic = config.n_critic
        self.grad_penalty_weight = config.g_penalty
        self.dataset_name = config.dataset
        self.total_images = total_images
        self.g_opt = ops.AdamOptWrapper(learning_rate=config.g_lr)
        self.d_opt = ops.AdamOptWrapper(learning_rate=config.d_lr)
        self.G = self.build_generator()
        self.D = self.build_discriminator()

        #self.G.summary()
        #self.D.summary()

    def get_models_weight(self):
        return self.G.get_weights(), self.D.get_weights()

    def set_models_weight(self, G_weight, D_weight):
        self.G.set_weights(G_weight)
        self.D.set_weights(D_weight)

    def train(self, dataset, group_ID):
        z = tf.constant(random.normal((config.n_samples, 1, 1, self.z_dim)))
        g_train_loss = metrics.Mean()
        d_train_loss = metrics.Mean()
        g_loss_result = 0
        d_loss_result = 0

        for epoch in range(self.epochs):
            bar = pbar(self.total_images, self.batch_size, epoch, self.epochs, group_ID)
            for batch in dataset:
                for _ in range(self.n_critic):
                    self.train_d(batch)
                    d_loss = self.train_d(batch)
                    d_train_loss(d_loss)

                g_loss = self.train_g()
                g_train_loss(g_loss)
                self.train_g()

                bar.postfix['g_loss'] = f'{g_train_loss.result():6.3f}'
                bar.postfix['d_loss'] = f'{d_train_loss.result():6.3f}'
                bar.update(self.batch_size)

            g_loss_result = g_train_loss.result()
            d_loss_result = d_train_loss.result()
            g_train_loss.reset_states()
            d_train_loss.reset_states()

            bar.close()
            del bar

            samples = self.generate_samples(z)
            # for emnist, squeeze should be removed
            image_grid = img_merge(samples, n_rows=8)
            save_image_grid(image_grid, epoch + 1)
        
        return g_loss_result.numpy(), d_loss_result.numpy()

    @tf.function
    def train_g(self):
        z = random.normal((self.batch_size, 1, 1, self.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            loss = ops.g_loss_fn(fake_logits)
        grad = t.gradient(loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss

    @tf.function
    def train_d(self, x_real):
        z = random.normal((self.batch_size, 1, 1, self.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            real_logits = self.D(x_real, training=True)
            cost = ops.d_loss_fn(fake_logits, real_logits)
            gp = self.gradient_penalty(partial(self.D, training=True), x_real, x_fake)
            cost += self.grad_penalty_weight * gp
        grad = t.gradient(cost, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grad, self.D.trainable_variables))
        return cost

    def gradient_penalty(self, f, real, fake):
        alpha = random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
        diff = fake - real
        inter = real + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = f(inter)
        grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp

    @tf.function
    def generate_samples(self, z):
        """Generates sample images using random values from a Gaussian distribution."""
        return self.G(z, training=False)

    def build_generator(self):
        if(config.dataset == 'emnist'): 
            data_dim = 1
        else: 
            data_dim = 3

        dim = self.image_size
        mult = dim // 8

        x = inputs = layers.Input((1, 1, self.z_dim))
        x = ops.UpConv2D(dim * mult, 4, 1, 'valid')(x)
        x = ops.BatchNorm()(x)
        x = layers.ReLU()(x)

        while mult > 1:
            x = ops.UpConv2D(dim * (mult // 2))(x)
            x = ops.BatchNorm()(x)
            x = layers.ReLU()(x)

            mult //= 2

        x = ops.UpConv2D(data_dim)(x)
        x = layers.Activation('tanh')(x)
        return models.Model(inputs, x, name='Generator')

    def build_discriminator(self):
        if(config.dataset == 'emnist'): 
            data_dim = 1
        else: 
            data_dim = 3
        dim = self.image_size
        mult = 1
        i = dim // 2

        x = inputs = layers.Input((dim, dim, data_dim))
        x = ops.Conv2D(dim)(x)
        x = ops.LeakyRelu()(x)

        while i > 4:
            x = ops.Conv2D(dim * (2 * mult))(x)
            x = ops.LayerNorm(axis=[1, 2, 3])(x)
            x = ops.LeakyRelu()(x)

            i //= 2
            mult *= 2

        x = ops.Conv2D(1, 4, 1, 'valid')(x)
        return models.Model(inputs, x, name='Discriminator')