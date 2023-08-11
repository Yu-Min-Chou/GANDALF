import asyncio
import os
import signal
import pickle
import numpy as np

from absl import flags

from sklearn import cluster, metrics
import matplotlib.pyplot as plt
import config

FLAGS = flags.FLAGS

def get_datainfo(id):

    inputfile = 'saved_ds_four/datainfo.data'
    fw = open(inputfile, 'rb')
    loaded = pickle.load(fw)
    dataset_dist = loaded['imbalance_dist']
    num_data = loaded['num_data']

    cli_dataset_dist = dataset_dist[id] / np.sum(dataset_dist[id])
    cli_dataset_size = np.sum(dataset_dist[id] * num_data).astype(int)

    return pickle.dumps({"ID": id, "dataset_dist": cli_dataset_dist, "dataset_size": cli_dataset_size}), cli_dataset_size

def k_cluster(dataset_dist):
    
    KM=cluster.KMeans(n_clusters=config.n_mediators,init='random',random_state=5)
    KM.fit(dataset_dist)

    return KM.predict(dataset_dist)