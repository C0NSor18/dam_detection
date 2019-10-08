#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:02:45 2019

@author: stephan
"""
#import tensorflow as tf

from scripts.experiment import run_experiment
#import tensorflow as tf
from tensorflow.keras.backend import clear_session
import os
# The reproducibility problem is in the GPU!!
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#tf.random.set_random_seed(3)
#tf.keras.backend.clear_session()
#tf.reset_default_graph()


#suppress information messages
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# the config gets converted to dict of values/lists
# tuples are converted to lists for some reason after the config
# is fetched from ex (and ex.add(config))
    
    

fit_params = {'lr': 0.0001,
			  'epochs': 2,
			  'reduce_lr_on_plateau': True,
			  'model': 'convnet'}

data_params = {'batch_size' :1,
			   'buffer_size':3000,
			   'use_augment': False,
			   'stretch_colorpsace': True,
			   'bridge_separate': True}

model_params = {'channels': ['B4', 'B3', 'B2'],
				'target_size': [257, 257],
				'num_classes': 3}


config = {'fit_params': fit_params,
		  'data_params': data_params,
		  'model_params': model_params}

run_experiment(config)

clear_session()