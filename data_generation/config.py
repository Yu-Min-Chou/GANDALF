# For emnist
dataset = 'emnist'
channel_size = 1

# # For cinic10
# dataset = 'cinic10'
# channel_size = 3

batch_size = 128
image_size = 32
num_client = 20

# setting of spliting dataset
mode = 'evaluation'
imbalanced_saved_path = '/split_datasets_' + dataset + '/' + mode + '/'
shared_saved_path = '/split_datasets_' + dataset + '/globally_shared/client'