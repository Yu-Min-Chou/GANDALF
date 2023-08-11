from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cinic10
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

    def cinic10_generate(self):
        XTrain, YTrain, XVal, YVal, XTest, YTest = cinic10.loadData("cinic10_raw")
        ds_train = tf.data.Dataset.from_tensor_slices((XTrain, YTrain))
        ds_test = tf.data.Dataset.from_tensor_slices((XTest,YTest))
        ds_train = ds_train.map(lambda x, y: self.preprocess_image(x,y), AUTOTUNE)
        ds_test = ds_test.map(lambda x, y: self.preprocess_image(x, y), AUTOTUNE)

        train_path = '/cinic10/train'
        test_path = '/cinic10/test'

        tf.data.experimental.save(ds_train, train_path)
        tf.data.experimental.save(ds_test, test_path)

        return

    def emnist_generate(self):
        ds, self.dataset_info = tfds.load(name='emnist/balanced',
                                        with_info=True,
                                        as_supervised=True)
        ds_train, ds_test = ds["train"], ds["test"]
        ds_train = ds_train.map(lambda x, y: self.preprocess_image(x,y), AUTOTUNE)
        ds_test = ds_test.map(lambda x, y: self.preprocess_image(x, y), AUTOTUNE)

        train_path = '/emnist/train'
        test_path = '/emnist/test'

        tf.data.experimental.save(ds_train, train_path)
        tf.data.experimental.save(ds_test, test_path)

        return

    def generate_imbalance_dataset(self):
        if(self.dataset_name != "cinic10"):
            ds, self.dataset_info = tfds.load(name='emnist/balanced',
                                            split='train',
                                            with_info=True,
                                            as_supervised=True)
            ds = ds.map(lambda x, y: self.preprocess_image(x,y), AUTOTUNE)
        else:
            path = '/cinic10/train'
            shape_x = (config.image_size, config.image_size, config.channel_size)
            shape_y = ()
            spec = (tf.TensorSpec(shape_x, dtype = tf.float32), tf.TensorSpec(shape_y, dtype = tf.int32))
            ds = tf.data.experimental.load(path, spec)

        self.get_imbalance_dataset(ds, config.mode)

        return

    def generate_shared_dataset(self, alpha=0.2, beta=0.2):
        if(self.dataset_name == 'cinic10'):
            path = '/cinic10/train'
            ds = tf.data.experimental.load(path)
            ds = ds.shuffle(50000, reshuffle_each_iteration=True)

            dataset_numpy = tfds.as_numpy(ds)
            num_class = 10
            globally_shared_X = []
            globally_shared_Y = []
            shared_X = []
            shared_Y = []
            tmp_num_list = []

            n_total_image = 90000
            print('n_total_image: {}'.format(n_total_image))

            n_sample_shared_dataset = int(n_total_image * beta)
            n_sample_per_class = int(n_sample_shared_dataset / num_class)
            n_sample_per_client = int(n_total_image * alpha * beta)
            print('n_sample_shared_dataset: {}'.format(n_sample_shared_dataset))
            print('n_sample_per_class: {}'.format(n_sample_per_class))
            print('n_sample_per_client: {}'.format(n_sample_per_client))

            for i in range(config.num_client):
                tmp_shared_X = np.zeros([n_sample_per_client, config.image_size, config.image_size, config.channel_size], dtype = np.float32)
                tmp_shared_Y = np.zeros([n_sample_per_client], dtype = np.int32)
                tmp_num_data = np.zeros([num_class], dtype = np.int32)
                shared_X.append(tmp_shared_X)
                shared_Y.append(tmp_shared_Y)
                tmp_num_list.append(tmp_num_data)

            index_record = np.zeros([config.num_client], dtype = np.int32)
            
            count = np.zeros([num_class], dtype = np.int32)
            indices = np.arange(n_sample_shared_dataset)

            for x, y in dataset_numpy:
                if(count[y] < n_sample_per_class):
                    globally_shared_X.append(x)
                    globally_shared_Y.append(y)
                    count[y] += 1

            
            for client in range(config.num_client):
                np.random.shuffle(indices)
                for index in indices:
                    if(tmp_num_list[client][globally_shared_Y[index]] < (n_sample_per_client / num_class)):
                        shared_X[client][index_record[client]] = globally_shared_X[index]
                        shared_Y[client][index_record[client]] = globally_shared_Y[index]
                        tmp_num_list[client][globally_shared_Y[index]] += 1
                        index_record[client] += 1
                    
            print(index_record) 

            for i in range(config.num_client):
                ds_shared = tf.data.Dataset.from_tensor_slices((shared_X[i], shared_Y[i]))
                path = config.shared_saved_path + str(i)
                tf.data.experimental.save(ds_shared, path)

            print('globally shared dataset has been stored')


        elif(self.dataset_name == 'emnist'):
            ds, self.dataset_info = tfds.load(name='emnist/balanced',
                                            split='train',
                                            with_info=True,
                                            as_supervised=True)
            ds = ds.map(lambda x, y: self.preprocess_image(x,y), AUTOTUNE)

            dataset_numpy = tfds.as_numpy(ds)
            num_class = 47
            globally_shared_X = []
            globally_shared_Y = []
            shared_X = []
            shared_Y = []
            tmp_num_list = []

            n_total_image = 112800
            print('n_total_image: {}'.format(n_total_image))

            n_sample_shared_dataset = int(n_total_image * beta)
            n_sample_per_class = int(n_sample_shared_dataset / num_class)
            n_sample_per_client = int(n_total_image * alpha * beta / num_class) * num_class
            print('n_sample_shared_dataset: {}'.format(n_sample_shared_dataset))
            print('n_sample_per_class: {}'.format(n_sample_per_class))
            print('n_sample_per_client: {}'.format(n_sample_per_client))

            for i in range(config.num_client):
                tmp_shared_X = np.zeros([n_sample_per_client, config.image_size, config.image_size, config.channel_size], dtype = np.float32)
                tmp_shared_Y = np.zeros([n_sample_per_client], dtype = np.int32)
                tmp_num_data = np.zeros([num_class], dtype = np.int32)
                shared_X.append(tmp_shared_X)
                shared_Y.append(tmp_shared_Y)
                tmp_num_list.append(tmp_num_data)

            index_record = np.zeros([config.num_client], dtype = np.int32)
            
            count = np.zeros([num_class], dtype = np.int32)
            indices = np.arange(n_sample_shared_dataset)

            for x, y in dataset_numpy:
                if(count[y] < n_sample_per_class):
                    globally_shared_X.append(x)
                    globally_shared_Y.append(y)
                    count[y] += 1

            
            for client in range(config.num_client):
                np.random.shuffle(indices)
                for index in indices:
                    if(tmp_num_list[client][globally_shared_Y[index]] < (n_sample_per_client / num_class)):
                        shared_X[client][index_record[client]] = globally_shared_X[index]
                        shared_Y[client][index_record[client]] = globally_shared_Y[index]
                        tmp_num_list[client][globally_shared_Y[index]] += 1
                        index_record[client] += 1
                    
            print(index_record) 

            for i in range(config.num_client):
                ds_shared = tf.data.Dataset.from_tensor_slices((shared_X[i], shared_Y[i]))
                path = config.shared_saved_path + str(i)
                tf.data.experimental.save(ds_shared, path)

            print('globally shared dataset has been stored')
            

        return

    def get_imbalance_dataset(self, dataset_train, mode='one_class', kl_required = 0.2):

        # Use KLD to evaluate how imbalance datasets are
        def KL(P, Q):
            """ Epsilon is used here to avoid conditional code for
            checking that neither P nor Q is equal to 0. """
            epsilon = 0.0000001

            P = P+epsilon
            Q = Q+epsilon

            divergence = np.sum(P*np.log(P/Q))
            return divergence


        # number of class of different dataset
        if config.dataset == 'emnist':
            num_class = 47
        else:
            num_class = 10

        num_clients = config.num_client
        uniform_ds = np.full(num_class, 1/num_class)
        imbalance_dist = np.zeros([num_clients, num_class], dtype=np.float32)

        # get distribution of imbalance datasets

        if mode[-5:] == 'class':

            if mode == 'one_class': class_each_client = 1
            elif mode == 'two_class': class_each_client = 2
            elif mode == 'four_class': class_each_client = 4

            assert(num_clients >= int(num_class/class_each_client))

            for client in range(num_clients):
                for i in range(class_each_client):
                    if(((client*class_each_client+i)%num_class) < ((num_clients*class_each_client)%num_class)):
                        part = 1 / math.ceil(num_clients * class_each_client / num_class)
                    else:
                        part = 1 / math.floor(num_clients * class_each_client / num_class)
                    imbalance_dist[client][(client*class_each_client+i)%num_class] = part  

        elif mode == 'evaluation':
            minor_class_ratio  = 0.01
            imbalance_dist = np.full([num_clients, num_class], minor_class_ratio, dtype=np.float32)
            class_each_client = 8
            for client in range(num_clients):
                for i in range(class_each_client):
                    if(((client*class_each_client+i)%num_class) < ((num_clients*class_each_client)%num_class)):
                        part = (1 - num_clients * minor_class_ratio) / math.ceil(num_clients * class_each_client / num_class)
                    else:
                        part = (1 - num_clients * minor_class_ratio) / math.floor(num_clients * class_each_client / num_class)
                    imbalance_dist[client][(client*class_each_client+i)%num_class] += part
            print(imbalance_dist)


        # get number of data of each label
        dataset_numpy = tfds.as_numpy(dataset_train)
        num_data = np.zeros([num_class], dtype=np.int32)

        for x, y in dataset_numpy:
            num_data[y] += 1

        # create list for imbalance datasets
        imbalance_X = []
        imbalance_Y = []
        tmp_num_list = []
        required_num_list = []
        for i in range(num_clients):
            required_num_data = imbalance_dist[i] * num_data
            required_num_data = required_num_data.astype(int)
            required_num_list.append(required_num_data)
            tmp_client_X = np.zeros([np.sum(required_num_data), config.image_size, config.image_size, config.channel_size], dtype=np.float32)
            tmp_client_Y = np.zeros([np.sum(required_num_data)], dtype=np.int32)
            tmp_num_data = np.zeros([num_class], dtype=np.int32)
            imbalance_X.append(tmp_client_X)
            imbalance_Y.append(tmp_client_Y)
            tmp_num_list.append(tmp_num_data)
            
            # print(i, imbalance_dist[i])
            print(np.sum(required_num_data))

        index_record = np.zeros((num_clients), dtype = np.int32)
        # append data to each client
        for x, y in dataset_numpy:
            for client in range(num_clients):
                if tmp_num_list[client][y] < required_num_list[client][y]:
                    tmp_num_list[client][y] += 1
                    imbalance_X[client][index_record[client]] = x
                    imbalance_Y[client][index_record[client]] = y
                    index_record[client] += 1
                    break
            
        # save dataset
        for i in range(num_clients):
            ds_imbalance = tf.data.Dataset.from_tensor_slices((imbalance_X[i], imbalance_Y[i]))
            ds_imbalance = ds_imbalance.shuffle(50000, reshuffle_each_iteration=True)
            path = config.imbalanced_saved_path + 'client' + str(i)
            tf.data.experimental.save(ds_imbalance, path)

        # dump datainfo
        datainfo = {"imbalance_dist": imbalance_dist, "num_data": num_data}
        path = config.imbalanced_saved_path + 'datainfo.data'
        fw = open(path, 'wb')
        pickle.dump(datainfo, fw)
        fw.close()

        print("mode: {}, saved_path = {}".format(config.mode, config.imbalanced_saved_path))

        return