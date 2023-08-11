from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from absl import app
from tensorflow import keras
from tensorflow.python.ops import control_flow_util

import config
import datasetpipeline

import pickle

keras.backend.clear_session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

import tensorflow as tf


def main(argv):
    del argv

    pipeline = datasetpipeline.DatasetPipeline()

    if(config.dataset == "cinic10"):
        pipeline.cinic10_generate()
    else if(config.dataset == "emnist"):
        pipeline.emnist_generate()

    pipeline.generate_imbalance_dataset()

    # pipeline.generate_shared_dataset()


if __name__ == '__main__':
    app.run(main)
