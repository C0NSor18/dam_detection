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
import os
import argparse
from pprint import pprint

# The reproducibility problem is in the GPU!!
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#tf.random.set_random_seed(3)
#tf.keras.backend.clear_session()
#tf.reset_default_graph()

argparser = argparse.ArgumentParser(
    description='run and train experiments')

argparser.add_argument('-m',  '--model', type=str, help='which model to use')
argparser.add_argument('--batch_size', default=32, type=int, help='mini batch size to use')
argparser.add_argument('--augmentations', default=True, type=bool, help='use augmentations (True, False)')
argparser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
argparser.add_argument('--sa', default=True, type=bool, help='use under/over sampling? (True, False)')
argparser.add_argument('--epoch', default=10, type=int, help='number of epochs (int)')
    
    

def _main_(args):
    # ======================
    # BASE PARAMETERS
    # ======================
    print(args)
    #os.nice(19)
    fit_params = {'model': args.model,
                  'lr': args.lr,
                  'epochs': args.epoch,
                  'reduce_lr_on_plateau': True}

    data_params = {'use_sampling': args.sa,
                   'batch_size' :args.batch_size,
                   'buffer_size':3000,
                   'augmentations': args.augmentations}

    model_params = {'channels': ['B4', 'B3', 'B2', 'AVE'],
                    'target_size': [257, 257],
                    'num_classes': 2,
                    'weights': 'imagenet'}


    # ======================
    # Darknet 19 detection
    # ======================

    config = {'fit_params': fit_params,
              'data_params': data_params,
              'model_params': model_params}

    

    run_experiment(config)

if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)

