from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import asyncio
import utils
import signal
import pickle
import argparse, sys
import numpy as np
import multiprocessing
from absl import app
from tensorflow import keras
from tensorflow.python.ops import control_flow_util
from nats.aio.client import Client as NATS
from diffprivlib.mechanisms.gaussian import Gaussian

from model import resnet_models
from model import gan_models
import config

keras.backend.clear_session()
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

def gan_augmentation(n_samples, ID, gan_model_ID):
    try:
        import tensorflow as tf
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ID%4)

        G_weight = []
        D_weight = []

        for i in range(len(gan_model_ID)):
            inputfile = os.path.join(config.gan_weight_path, 'gan_weight{}'.format(gan_model_ID[i]))
            fw = open(inputfile, 'rb')
            loaded = pickle.load(fw)
            G_weight.append(loaded['G_weight'])
            D_weight.append(loaded['D_weight'])

        wgangp = gan_models.WGANGP(0)
        samples = []

        for i in range(len(gan_model_ID)):
            if n_samples[i] <= 0:
                continue
            wgangp.set_models_weight(G_weight[i], D_weight[i])
            z = tf.constant(tf.random.normal((n_samples[i], 1, 1, config.z_size)))
            fake_samples = (tf.cast(wgangp.generate_samples(z), tf.float32)).numpy()
            samples.append(fake_samples)

        return samples

    except Exception as e:
        print(e)

def labeling(ID, samples):
    try:
        import tensorflow as tf
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ID%4)

        inputfile = os.path.join(config.labeling_weight_path, config.dataset)
        fw = open(inputfile, 'rb')
        loaded = pickle.load(fw)
        resnet_weight = loaded['weights']

        resnet = resnet_models.Resnet()
        resnet.set_model_weight(resnet_weight)
        predict_result = []
        label = []
        k = 0

        for i in range(len(samples)):
            predict_result.append((resnet.predict(samples[i])).numpy())
            tmp = np.zeros(predict_result[i].shape[0], dtype = np.int32)
            for j in range(predict_result[i].shape[0]):
                tmp[j] = np.argmax(predict_result[i][j])
            label.append(tmp)

        return label

    except Exception as e:
        print('labeling: ', e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def adjustment(ID, n_samples, gan_dist, fake_samples, fake_labels, orginal_dataset_dist, orginal_dataset_size):
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ID%4)

        n_class = gan_dist[0].shape[0]
        aug_factor = config.aug_factor
        # number of data theoretically generated
        n_theoretical_data = ((np.max(orginal_dataset_dist) - orginal_dataset_dist) * orginal_dataset_size).astype(np.int32) * aug_factor
        print("n_theoretical_data: {}".format(n_theoretical_data))

        # number of data actually generated
        n_actual_data = np.zeros((n_class))
        for i in range(len(fake_labels)):
            for j in range(len(fake_labels[i])):
                n_actual_data[fake_labels[i][j]] += 1

        n_actual_data = n_actual_data.astype(int)
        print("n_actual_data: {}".format(n_actual_data))

        # replace fake data with real data
        if(config.replace_fake_with_real):
            path = config.whole_dataset_path
            ds_train = tf.data.experimental.load(path)
            ds_train = ds_train.shuffle(50000, reshuffle_each_iteration=True)
            ds_train_numpy = tfds.as_numpy(ds_train)
            tmp_X = np.zeros([sum(n_actual_data), config.image_size, config.image_size, config.channel_size], dtype = np.float32)
            tmp_Y = np.zeros([sum(n_actual_data)], dtype = np.int32)
            label_count = np.zeros((n_class), dtype = np.int32)
            index_count = 0

            for x, y in ds_train_numpy:
                if(label_count[y] < n_actual_data[y]):
                    tmp_X[index_count] = x
                    tmp_Y[index_count] = y
                    label_count[y] += 1
                    index_count += 1

            fake_samples = []
            fake_labels = []
            fake_samples.append(tmp_X)
            fake_labels.append(tmp_Y)


        n_adjust = n_theoretical_data - n_actual_data
        print('n_adjust: {}'.format(n_adjust))
        # adjustment
        # perform traditional augmentation
        path = config.original_dataset_path + str(ID) 
        dataset = tf.data.experimental.load(path)

        if(config.static_augmentation):
            data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 10,
                                                                            horizontal_flip = True,
                                                                            width_shift_range = 0.05,
                                                                            height_shift_range = 0.05)

            tmp_dataset = dataset.batch(config.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
            data_aug = []
            for x, y in tmp_dataset:
                x_aug, y_aug = data_generator.flow(x, y, batch_size = config.batch_size).next()
                for i in range(config.batch_size):
                    if(n_adjust[y_aug[i]] > 0):
                        data_aug.append((x_aug[i], y_aug[i]))
                        n_adjust[y_aug[i]] -= 1
        
            data_aug_x = np.zeros((len(data_aug), config.image_size, config.image_size, config.channel_size), dtype = np.float32)
            data_aug_y = np.zeros((len(data_aug)), dtype = np.int32)

            for i in range(len(data_aug)):
                data_aug_x[i], data_aug_y[i] = data_aug[i]

            dataset_aug = tf.data.Dataset.from_tensor_slices((data_aug_x, data_aug_y))
            dataset = dataset.concatenate(dataset_aug)
        
        # delete redundant fake data
        n_total_fake_samples = 0
        for i in range(len(n_samples)):
            if n_samples[i] > 0:
                n_total_fake_samples += n_samples[i]

        fake_samples_con = np.zeros((n_total_fake_samples, config.image_size, config.image_size, config.channel_size), dtype = np.float32)
        fake_labels_con = np.zeros((n_total_fake_samples,), dtype = np.int32)

        idx = 0
        for i in range(len(fake_samples)):
            fake_samples_con[idx: idx + fake_samples[i].shape[0]] = fake_samples[i]
            fake_labels_con[idx: idx + fake_labels[i].shape[0]] = fake_labels[i]
            idx += fake_samples[i].shape[0]

        
        idx_delete = []
        for i in range(fake_samples_con.shape[0]):
            if(n_adjust[fake_labels_con[i]] < 0):
                idx_delete.append(i)
                n_adjust[fake_labels_con[i]] += 1
        np_idx_delete = np.array(idx_delete)
        fake_samples_con = np.delete(fake_samples_con, np_idx_delete, 0)
        fake_labels_con = np.delete(fake_labels_con, np_idx_delete, 0)

        dataset_fake = tf.data.Dataset.from_tensor_slices((fake_samples_con, fake_labels_con))

        dataset = dataset.concatenate(dataset_fake)
        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)
        dataset = dataset.batch(config.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
        path = config.gan_aug_dataset_path + str(ID)
        tf.data.experimental.save(dataset, path)

        n_orginal_data = (orginal_dataset_dist * orginal_dataset_size).astype(np.int32)
        
        np_dataset = tfds.as_numpy(dataset)
        n_aug_data = np.zeros((n_class))

        for x, y in np_dataset:
            for i in range(config.batch_size):
                n_aug_data[y[i]] += 1

        aug_dataset_size = np.sum(n_aug_data).astype(np.int32)
        print('aug_dataset_size: {}'.format(aug_dataset_size))
        print("number of aug data:     {}".format(n_aug_data))

        aug_dataset_dist = n_aug_data / aug_dataset_size

        return aug_dataset_dist, aug_dataset_size

    except Exception as e:
        print('adjustment: ', e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def train(C_weight, ID, group_ID):
    try:
        import tensorflow as tf
        from tensorflow.python.keras import layers
        os.environ["CUDA_VISIBLE_DEVICES"] = str(group_ID)
        data_augmentation = tf.keras.Sequential([
                layers.ZeroPadding2D(padding=2),
                layers.RandomCrop(32, 32),
                layers.RandomFlip('horizontal'),
            ])
        data_augmentation.build(input_shape=(None, config.image_size, config.image_size, config.channel_size))

        # load augmented_dataset from disk
        if(config.gan_augmentation):
            path = config.gan_aug_dataset_path + str(ID)
        elif(config.shared_augmentation):
            path = config.shared_aug_dataset_path + str(ID)
        elif(config.only_static_augmentation):
            path = config.static_aug_dataset_path + str(ID)
        else:
            print("You should use one of the dataset: gan_aug, shared_aug, static_aug")

        dataset = tf.data.experimental.load(path)
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

        resnet = resnet_models.Resnet()
        resnet.set_model_weight(C_weight)

        c_loss_result, c_accu_result = resnet.train(dataset)

        return resnet.get_model_weight(), c_loss_result, c_accu_result
        
    except Exception as e:
        print('train: ', e)

def shared_augmentation(ID, group_ID):
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ID%4)

        if(config.dataset == 'cinic10'):
            n_class = 10
        elif(config.dataset == 'emnist/bymerge'):
            n_class = 47

        path = config.original_dataset_path + str(ID) 
        dataset = tf.data.experimental.load(path)

        path = config.shared_dataset_path + str(ID)
        dataset_shared = tf.data.experimental.load(path)

        dataset = dataset.concatenate(dataset_shared)
        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)
        dataset = dataset.batch(config.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
        path = config.shared_aug_dataset_path + str(ID)
        tf.data.experimental.save(dataset, path)
        
        np_dataset = tfds.as_numpy(dataset)
        n_aug_data = np.zeros((n_class))

        for x, y in np_dataset:
            for i in range(config.batch_size):
                n_aug_data[y[i]] += 1

        aug_dataset_size = np.sum(n_aug_data).astype(np.int32)
        print("number of aug data:     {}".format(n_aug_data))

        aug_dataset_dist = n_aug_data / aug_dataset_size

        return aug_dataset_dist, aug_dataset_size

        
    except Exception as e:
        print('shared_augmentation: ', e)

def only_static_augmentation(ID, group_ID, orginal_dataset_dist, orginal_dataset_size):
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ID%4)

        if(config.dataset == 'cinic10'):
            n_class = 10
        elif(config.dataset == 'emnist'):
            n_class = 47

        aug_factor = config.aug_factor
        # number of data theoretically generated
        n_theoretical_data = ((np.max(orginal_dataset_dist) - orginal_dataset_dist) * orginal_dataset_size).astype(np.int32) * aug_factor
        print(n_theoretical_data)

        path = config.original_dataset_path + str(ID) 
        dataset = tf.data.experimental.load(path)

        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 10,
                                                                            horizontal_flip = False,
                                                                            width_shift_range = 0.01,
                                                                            height_shift_range = 0.01)

        tmp_dataset = dataset.batch(config.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
        data_aug = []
        for x, y in tmp_dataset:
            x_aug, y_aug = data_generator.flow(x, y, batch_size = config.batch_size).next()
            for i in range(config.batch_size):
                if(n_theoretical_data[y_aug[i]] > 0):
                    data_aug.append((x_aug[i], y_aug[i]))
                    n_theoretical_data[y_aug[i]] -= 1

        print(len(data_aug))
        
        data_aug_x = np.zeros((len(data_aug), config.image_size, config.image_size, config.channel_size), dtype = np.float32)
        data_aug_y = np.zeros((len(data_aug)), dtype = np.int32)

        for i in range(len(data_aug)):
            data_aug_x[i], data_aug_y[i] = data_aug[i]

        dataset_aug = tf.data.Dataset.from_tensor_slices((data_aug_x, data_aug_y))
        dataset = dataset.concatenate(dataset_aug)
        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)
        dataset = dataset.batch(config.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
        path = config.static_aug_dataset_path + str(ID)
        tf.data.experimental.save(dataset, path)
        
        np_dataset = tfds.as_numpy(dataset)
        n_aug_data = np.zeros((n_class))

        for x, y in np_dataset:
            for i in range(config.batch_size):
                n_aug_data[y[i]] += 1

        aug_dataset_size = np.sum(n_aug_data).astype(np.int32)
        print("number of aug data:     {}".format(n_aug_data))

        aug_dataset_dist = n_aug_data / aug_dataset_size

        return aug_dataset_dist, aug_dataset_size

        
    except Exception as e:
        print('only_static_augmentation: ', e)

class Client_cb():
    def __init__(self, loop, nc, socket):
        self.loop = loop
        self.nc = nc
        self.socket = socket
        self.dataset = None
        self.ID = -1
        self.group_ID = -1
        self.gan_model_ID = -1
        self.dataset_dist = None
        self.dataset_size = None
        self.gan_dist = None
        self.gpu_used = False
        self.fake_samples = None
        self.fake_labels = None
        self.C_weight = None
        self.c_loss_result = None
        self.c_accu_result = None
        

    def get_id(self):
        return self.ID

    def get_group_id(self):
        return self.group_ID

    def get_dist(self):
        DP = Gaussian(epsilon=config.dist_epsilon, delta=config.dist_delta, sensitivity=config.dist_sensitivity)
        randomise_dataset_dist = np.zeros(self.dataset_dist.shape)
        for i in range(len(randomise_dataset_dist)):
            randomise_dataset_dist[i] = DP.randomise(self.dataset_dist[i])
            if(randomise_dataset_dist[i] > 1.0):
                randomise_dataset_dist[i] = 1.0
            elif(randomise_dataset_dist[i] < 0.0):
                randomise_dataset_dist[i] = 0.0
        return randomise_dataset_dist

    def load_datainfo(self):
        inputfile = config.datainfo_path
        fw = open(inputfile, 'rb')
        loaded = pickle.load(fw)
        dataset_dist = loaded['imbalance_dist']
        num_data = loaded['num_data']

        self.dataset_dist = (dataset_dist[self.ID] * num_data) / np.sum(dataset_dist[self.ID] * num_data)
        self.dataset_size = np.sum(dataset_dist[self.ID] * num_data).astype(int)
        print(self.dataset_size)

    async def create_gan_process(self, n_samples):
        try:
            pool = multiprocessing.get_context('spawn').Pool(1)
            result = pool.starmap(gan_augmentation, [(n_samples, self.ID, self.gan_model_ID)])
            self.fake_samples = result[0]
            pool.close()
            pool.join()

        except Exception as e:
            print('create_gan_process: ', e)

    async def create_label_process(self):
        try:
            pool = multiprocessing.get_context('spawn').Pool(1)
            result = pool.starmap(labeling, [(self.ID, self.fake_samples)])
            self.fake_labels = result[0]
            pool.close()
            pool.join()

        except Exception as e:
            print('create_label_process: ', e)

    async def create_adjustment_process(self, n_samples):
        try:
            pool = multiprocessing.get_context('spawn').Pool(1)
            result = pool.starmap(adjustment, [(self.ID, n_samples, self.gan_dist, self.fake_samples, self.fake_labels, self.dataset_dist, self.dataset_size)])
            self.dataset_dist = result[0][0]
            self.dataset_size = result[0][1]
            pool.close()
            pool.join()

        except Exception as e:
            print('create_adjustment_process: ', e)

    async def create_training_process(self):
        try:
            pool = multiprocessing.get_context('spawn').Pool(1)
            result = pool.starmap(train, [(self.C_weight, self.ID, self.group_ID)])
            self.C_weight = result[0][0]
            self.c_loss_result = result[0][1]
            self.c_accu_result = result[0][2]
            pool.close()
            pool.join()

        except Exception as e:
            print('create_training_process: ', e)

    async def create_shared_process(self):
        try:
            pool = multiprocessing.get_context('spawn').Pool(1)
            result = pool.starmap(shared_augmentation, [(self.ID, self.group_ID)])
            self.dataset_dist = result[0][0]
            self.dataset_size = result[0][1]
            pool.close()
            pool.join()

        except Exception as e:
            print('create_shared_process: ', e)

    async def create_only_static_process(self):
        try:
            pool = multiprocessing.get_context('spawn').Pool(1)
            result = pool.starmap(only_static_augmentation, [(self.ID, self.group_ID, self.dataset_dist, self.dataset_size)])
            self.dataset_dist = result[0][0]
            self.dataset_size = result[0][1]
            pool.close()
            pool.join()

        except Exception as e:
            print('create_only_static_process: ', e)

    async def gpu_used_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            finished_ID = loaded['ID']
            if((self.ID - config.n_gpus) == finished_ID):
                self.gpu_used = True

        except Exception as e:
            print(e)
        
    async def request_id_cb(self, msg):
        # receive a message from [request_id-reply]
        # message is a int, ID of this client
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            if(loaded['message'] == 'success'):
                print("get ID: ", loaded['ID'])
                self.ID = loaded['ID']
                self.load_datainfo()
            else:
                print("Amount of clients has been reached the upper limit")
                await self.socket.terminate()
                await asyncio.sleep(5)
        except Exception as e:
            print(e)

    async def request_gan_weight_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            gan_model_ids = loaded['gan_model_ids']
            gan_dist = loaded['gan_dist']
            self.gan_model_ID = gan_model_ids
            self.gan_dist = gan_dist

            print(self.dataset_dist)

            if(config.gan_augmentation):
                if(self.ID < config.n_gpus):
                    self.gpu_used = True

                while(not(self.gpu_used)):
                    print('client{} is waiting for available gpu...'.format(self.ID))
                    await asyncio.sleep(5)

                # calculate how many sample gan should generate
                aug_factor = config.aug_factor
                n_gan = len(self.gan_model_ID)
                n_class = self.dataset_dist.shape[0]

                a = np.zeros((n_gan, n_class))
                for i in range(n_gan):
                    a[i, :] = self.gan_dist[self.gan_model_ID[i]]

                a = a.transpose()
                b = (np.full((self.dataset_dist.shape), np.max(self.dataset_dist), dtype=np.float64) - self.dataset_dist) * aug_factor
                
                aug_weight = np.linalg.pinv(a).dot(b)
                n_samples = (self.dataset_size * aug_weight).astype(int)
                print(n_samples)

                print('Starting gan_augmentation')

                await self.gan_augmentation(n_samples)

                print('Starting labeling')

                await self.labeling()

                print('Starting adjustment')

                await self.adjustment(n_samples)

                print('client{} finish augmentation'.format(self.ID))

                # let next client use gpu
                target_subject = 'gpu_used'
                message_bytes = pickle.dumps({'ID': self.ID})
                await self.socket.publish(target_subject, message_bytes)
            elif(config.shared_augmentation):
                if(self.ID < config.n_gpus):
                    self.gpu_used = True

                while(not(self.gpu_used)):
                    print('client{} is waiting for available gpu...'.format(self.ID))
                    await asyncio.sleep(5)

                print('Starting shared_augmentation')

                await self.shared_augmentation()

                 # let next client use gpu
                target_subject = 'gpu_used'
                message_bytes = pickle.dumps({'ID': self.ID})
                await self.socket.publish(target_subject, message_bytes)
            elif(config.only_static_augmentation):
                if(self.ID < config.n_gpus):
                    self.gpu_used = True

                while(not(self.gpu_used)):
                    print('client{} is waiting for available gpu...'.format(self.ID))
                    await asyncio.sleep(5)

                print('Starting only_static_augmentation')

                await self.only_static_augmentation()

                 # let next client use gpu
                target_subject = 'gpu_used'
                message_bytes = pickle.dumps({'ID': self.ID})
                await self.socket.publish(target_subject, message_bytes)

            # send dataset information to server
            print(self.dataset_dist)
            message_bytes = pickle.dumps({'ID': self.ID, 'dataset_dist': self.dataset_dist, 'dataset_size': self.dataset_size})
            await self.socket.request('request_datainfo', message_bytes, self.request_datainfo_cb)


        except Exception as e:
            print(e)

    async def gan_augmentation(self, n_samples):
        # use multiprocessing
        task = self.loop.create_task(self.create_gan_process(n_samples))
        await task

        for i in range(len(self.fake_samples)):
            print('generate data shape: {}, type: {}'.format(self.fake_samples[i].shape, type(self.fake_samples[i])))


    async def labeling(self):
        # use multiprocessing
        task = self.loop.create_task(self.create_label_process())
        await task

    async def adjustment(self, n_samples):
        # use multiprocessing
        task = self.loop.create_task(self.create_adjustment_process(n_samples))
        await task

    async def shared_augmentation(self):
        task = self.loop.create_task(self.create_shared_process())
        await task

    async def only_static_augmentation(self):
        task = self.loop.create_task(self.create_only_static_process())
        await task

    async def request_datainfo_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            if(loaded['message'] == 'success'):
                print("client {}: request_info success".format(self.ID))
            else:
                print("client {}: request_info fail".format(self.ID))

        except Exception as e:
            print(e)

    async def publish_groupinfo_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            client_info = loaded['client_info']
            group_info = loaded['group_info']

            for i in range(len(client_info)):
                if(client_info[i] == self.ID):
                    self.group_ID = group_info[i]
                    print("Client{}, group ID: {}".format(self.ID, self.group_ID))
                    break

        except Exception as e:
            print('publish_groupinfo: ', e)

    async def request_check_network_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']

            message_bytes = pickle.dumps({'message': 'success'})
            await self.socket.publish(msg.reply, message_bytes)

        except Exception as e:
            print('request_check_network_cb:', e)

    async def train_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            C_weight = loaded['C_weight']

            self.C_weight = C_weight

            print('client{} receive weight, starting training...'.format(self.ID))

            await asyncio.sleep(1)

            task = self.loop.create_task(self.create_training_process())
            await task

            message_send = 'from client{}, '.format(self.ID)
            message_bytes = pickle.dumps({'message': message_send, 'loss': self.c_loss_result, 'accu': self.c_accu_result})

            target_subject = 'request_pass_weight_{}'.format(self.group_ID)
            await self.socket.request(target_subject, message_bytes, self.request_pass_weight_cb)

        except Exception as e:
            print('train_cb', e)

    async def request_pass_weight_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            next_client_id = loaded['next_client_id']

            message_send = 'from client{}'.format(self.ID)

            # DP for model weight
            DP = Gaussian(epsilon=config.weight_epsilon, delta=config.weight_delta, sensitivity=config.weight_sensitivity)
            np_C_weight = np.asarray(self.C_weight, dtype=object)
            for weight_list in np_C_weight:
                for x in np.nditer(weight_list, flags=['refs_ok'], op_flags=['readwrite']):
                    x[...] = np.array(DP.randomise(float(x)))
            self.C_weight = np_C_weight.tolist()

            weights_bytes = pickle.dumps({'message': message_send, 'C_weight': self.C_weight})

            if(message == 'client'):
                target_subject = 'train_client{}'.format(next_client_id)
            elif(message == 'mediator'):
                target_subject = 'train_client_to_mediator{}'.format(self.group_ID)

            await self.socket.publish(target_subject, weights_bytes)

        except Exception as e:
            print("request_pass_weight: ", e)

    async def terminate_process_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            print(message)

            await self.socket.terminate()

        except Exception as e:
            print('terminate_process_cb: ', e)
