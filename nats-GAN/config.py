servers = 'nats://127.0.0.1:4222'
# For emnist
# dataset = 'emnist'
# channel_size = 1

# For cinic10
dataset = 'cinic10'
channel_size = 3

# Hyperparameter of local training
local_epochs = 3
batch_size = 128
image_size = 32

z_size = 128
g_lr = .0001
d_lr = .00005
output_dir = './image'
g_penalty = 10.0
n_critic = 2
n_samples = 64

# Hyperparameter of Federated Learning
n_clients = 20
n_mediators = 2
min_rounds = 120
max_rounds = 150
n_converged = 10
converged_threshold = 0.2

# Differential privacy setting for local data distributions
dist_epsilon = 0.9
dist_delta = 0.9
dist_sensitivity = 0.00001

# Differential privacy setting for model weights
weight_epsilon = 0.33
weight_delta = 1.0
weight_sensitivity = 0.01

# Path of local datasets and trained GAN models
datainfo_path = '../the path of local datasets/datainfo.data'
ds_path = '../the path of local datasets/client'
saved_gan_weight_path = '../the path you want to store the trained GAN model'

