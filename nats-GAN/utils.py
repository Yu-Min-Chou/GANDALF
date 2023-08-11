import asyncio
import os
import signal
import pickle
import numpy as np
from diffprivlib.mechanisms.gaussian import Gaussian

from absl import flags

from sklearn import cluster, metrics
import matplotlib.pyplot as plt
import config

FLAGS = flags.FLAGS

def get_datainfo(id):

    inputfile = config.datainfo_path
    fw = open(inputfile, 'rb')
    loaded = pickle.load(fw)
    dataset_dist = loaded['imbalance_dist']
    num_data = loaded['num_data']

    cli_dataset_dist = dataset_dist[id] / np.sum(dataset_dist[id])
    cli_dataset_size = np.sum(dataset_dist[id] * num_data).astype(int)
    
    DP = Gaussian(epsilon=config.dist_epsilon, delta=config.dist_delta, sensitivity=config.dist_sensitivity)
    for i in range(len(cli_dataset_dist)):
        cli_dataset_dist[i] = DP.randomise(cli_dataset_dist[i])
        if(cli_dataset_dist[i] > 1.0):
            cli_dataset_dist[i] = 1.0
        elif(cli_dataset_dist[i] < 0.0):
            cli_dataset_dist[i] = 0.0

    return pickle.dumps({"ID": id, "dataset_dist": cli_dataset_dist, "dataset_size": cli_dataset_size}), cli_dataset_size

def k_cluster(dataset_dist):
    
    seed = 0
    while True:
        print("grouping with K-Means")
        KM=cluster.KMeans(n_clusters=config.n_mediators,init='random',random_state=seed)
        KM.fit(dataset_dist)
        count = np.zeros(config.n_mediators)
        result = KM.predict(dataset_dist)

        for i in result:
            count[i] += 1
        print(count)
        
        if((count >= (config.n_clients/config.n_mediators - 2)).all()):
            break
        else:
            seed += 1
            continue

    return KM.predict(dataset_dist)