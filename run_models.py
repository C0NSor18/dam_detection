#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:02:45 2019

@author: stephan
"""
#import tensorflow as tf

from scripts.experiment import run_experiment
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from generators.augmentations import rotate, flip
import os
# The reproducibility problem is in the GPU!!
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#tf.random.set_random_seed(3)
#tf.keras.backend.clear_session()
#tf.reset_default_graph()


# the config gets converted to dict of values/lists
# tuples are converted to lists for some reason after the config
# is fetched from ex (and ex.add(config))

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
# ======================
# BASE PARAMETERS
# ======================

#os.nice(19)
fit_params = {'model': None,
			  'lr': 0.0001,
			  'epochs': 15,
			  'reduce_lr_on_plateau': True}

data_params = {'use_sampling': True,
			   'batch_size' :32,
			   'buffer_size':3000,
			   'augmentations': [],
			   'stretch_colorpsace': True,
			   'bridge_separate': True}

model_params = {'channels': ['B4', 'B3', 'B2', 'NDWI', 'AVE'],
				'target_size': [128, 128],
				'num_classes': 3}

# ======================
# Fully convolutional net
# ======================
#fit_params['model'] = 'fcn'
#config = {'fit_params': fit_params,
#		  'data_params': data_params,
#		  'model_params': model_params}
#
#run_experiment(config)
#clear_session() 

# ======================
# Dilated FCN
# ======================
#fit_params['model'] = 'convnet'
## Requires exactly 61, 61 patches
#model_params['target_size'] = [61,61]
#config = {'fit_params': fit_params,
#		  'data_params': data_params,
#		  'model_params': model_params}
#
#run_experiment(config)
#clear_session() 

# ======================
# DenseNet
# ======================
#fit_params['model'] = 'densenet121'
#config = {'fit_params': fit_params,
#		  'data_params': data_params,
#		  'model_params': model_params}
#
#run_experiment(config)
#clear_session() 


# ======================
# ResNet50
# ======================
fit_params['model'] = 'resnet50'
config = {'fit_params': fit_params,
		  'data_params': data_params,
		  'model_params': model_params}

run_experiment(config)
clear_session() 


# ======================
# Convnet (dense layers)
# ======================
#fit_params['model'] = 'convnet'
#config = {'fit_params': fit_params,
#		  'data_params': data_params,
#		  'model_params': model_params}
#
#run_experiment(config)
#clear_session() 


