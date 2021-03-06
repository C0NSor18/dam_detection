#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 01:14:36 2019

@author: stephan
"""
import tensorflow as tf
import numpy as np
from scripts.constants import SEED
import os
from generators.augmentations import apply_augmentation
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


# bridges are not incorporated yet
NUM_CLASSES = 2
# TF parsing functions
def parse_serialized_example(example_proto):
    ''' Parser function
    Useful for functional extraction, i.e. .map functions
    
    Args:
        example_proto: a serialized example
        
    Returns:
        A dictionary with features, cast to float32
        This returns a dictionary of keys and tensors to which I apply the transformations.
    '''
    # feature columns of interest
    featuresDict = {
        'AVE': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # Elevation
        'B2': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # B
        'B3': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # G
        'B4': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # R
        'label': tf.io.FixedLenFeature([1], dtype=tf.float32), #label
        'NDWI': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # water index
        'MNDWI': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # modified water index
        'AWEISH': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # more water indices
        'AWEINSH':tf.io.FixedLenFeature([257, 257], dtype=tf.float32)
    }
    
    return tf.io.parse_single_example(example_proto, featuresDict)


def tf_stretch_image_colorspace(img):
    max_val = tf.reduce_max(img)
    return tf.cast(tf.divide(img, max_val), tf.float32)


#using a closure so we can add extra params to the map function from tf.Dataset
def parse_image(target_size, channels):
    ''' Stack individual RGB bands into a N dimensional array
    The RGB bands are still separate 1D arrays in the TFRecords, combine them into a single 3D array
    
    Args:
        features: A dictionary with the features (RGB bands or other channels that need to be concatenated)
    '''
    
    # print("using the general image parsing function")
    def parse_image_fun(features):
        #channels = list(features.values())
        label = features['label']
        
        # Get the image channels, and NDWI/AVE channels separately
        # we cannot import them all at once since they need separate preprocessing steps
        img_chan = [features[x] for x in channels if x in ['B4', 'B3', 'B2']]
        ndwi_chan = [features[x] for x in channels if x in ['NDWI']]
        mndwi_chan = [features[x] for x in channels if x in ['MNDWI']]
        aweish_chan = [features[x] for x in channels if x in ['AWEISH']]
        aweinsh_chan = [features[x] for x in channels if x in ['AWEINSH']]
        ave_chan = [features[x] for x in channels if x in ['AVE']]
    
        
        # stack the individual arrays, remove all redundant dimensions of size 1, and transpose them into the right order
        # (batch size, H, W, channels)
        img = tf.transpose(tf.squeeze(tf.stack(img_chan)))
        
        # stretch color spaces of the RGB channels
        img = tf_stretch_image_colorspace(img)
        
        # concatenate ndwi channel
        if ndwi_chan:
            # further normalization? -> NDWI [-1,1]
            ndwi_chan = tf.divide(tf.add(ndwi_chan, 1), 1 + 1)
            img = tf.concat([img, tf.transpose(ndwi_chan)], axis= 2)
            
        if mndwi_chan:  
            # Range MNDWI: [-1,1]
            mndwi_chan = tf.divide(tf.add(mndwi_chan, 1), 1 + 1)
            img = tf.concat([img, tf.transpose(mndwi_chan)], axis= 2)
            
        if aweish_chan:
            # Range Current implementation: unknown
            img = tf.concat([img, tf.transpose(aweish_chan)], axis= 2)
            
        if aweinsh_chan:
            # Range Current implementation: unknown
            img = tf.concat([img, tf.transpose(aweinsh_chan)], axis= 2)
            
        if ave_chan:
            # Range normalization: https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V1_1
            ave_chan = tf.divide(tf.add(ave_chan, 479) ,479 + 8859)
            img = tf.concat([img, tf.transpose(ave_chan)], axis= 2)

        
        # Additionally, resize the images to a desired size
        img = tf.image.resize(img, target_size)
		
		# label separator, when bridges need to be class 0, or 2
        # Below is legacy code: bridges have label "2", this casts them to 0
        label = tf.unstack(tf.cast(label, dtype = tf.int32))
        x = tf.constant(2)
        def f1(): return tf.add(label, 0)
        def f2(): return tf.add(label, -2)
        r = tf.cond(tf.less(label[0], x), f1, f2)
        label= tf.reduce_max(tf.one_hot(tf.cast(r, dtype=tf.int32), 2, dtype=tf.int32), axis=0)
        return img, label
    
    return parse_image_fun


# https://github.com/kmkolasinski/deep-learning-notes/blob/master/tf-oversampling/oversampling-datasets-example.ipynb
def undersampling_filter(probs, class_target_prob, undersampling_coef = 0.9):
	# higher undersampling_coef (a>1)  leads to heavier undersampling,  (a<1) leads to lessened undersampling
    def wrapper(example):
        """
        Computes if given example is rejected or not.
        """
        label = example['label']
        label = tf.unstack(tf.cast(label, dtype = tf.int32))  

        dam_label = tf.constant(1)
        #other_label = tf.constant(0)
        
        def other_fun(): return probs['other']
        def dam_fun(): return probs['dams']
		
        '''
        if bridge_sep_label:
            def bridge_fun(): return probs['bridges']
            def cond2(): return tf.cond(tf.equal(label[0], other_label), other_fun, bridge_fun)
            class_prob = tf.cond(tf.equal(label[0], dam_label), dam_fun, cond2)
        '''
        class_prob = tf.cond(tf.equal(label[0], dam_label), dam_fun, other_fun)

        prob_ratio = tf.cast(class_target_prob/class_prob, dtype=tf.float32)
        prob_ratio = prob_ratio ** undersampling_coef
        prob_ratio = tf.minimum(prob_ratio, 1.0)

        acceptance = tf.less_equal(tf.random_uniform([], dtype=tf.float32, seed=SEED), prob_ratio)
        # predicate must return a scalar boolean tensor
        return acceptance

    return wrapper


def oversampling_filter(probs, class_target_prob, oversampling_coef =0.9):
    
    def wrapper(example):
        """
        Computes if given example is rejected or not.
        """
        label = example['label']
        label = tf.unstack(tf.cast(label, dtype = tf.int32))  

        dam_label = tf.constant(1)
        # legacy: other_label = tf.constant(0)
        
        def other_fun(): return probs['other']
        def dam_fun(): return probs['dams']
		
        '''
		# legacy code from bridge separate
        if bridge_sep_label:
            def bridge_fun(): return probs['bridges']
            def cond2(): return tf.cond(tf.equal(label[0], other_label), other_fun, bridge_fun)
            class_prob = tf.cond(tf.equal(label[0], dam_label), dam_fun, cond2)
			
		'''

        class_prob = tf.cond(tf.equal(label[0], dam_label), dam_fun, other_fun)
        
        prob_ratio = tf.cast(class_target_prob/class_prob, dtype=tf.float32)
        # soften ratio is oversampling_coef==0 we recover original distribution
        prob_ratio = prob_ratio ** oversampling_coef 
        # for classes with probability higher than class_target_prob we
        # want to return 1
        prob_ratio = tf.maximum(prob_ratio, 1) 
        # for low probability classes this number will be very large
        repeat_count = tf.floor(prob_ratio)
        # prob_ratio can be e.g 1.9 which means that there is still 90%
        # of change that we should return 2 instead of 1
        repeat_residual = prob_ratio - repeat_count # a number between 0-1
        residual_acceptance = tf.less_equal(
                            tf.random_uniform([], dtype=tf.float32, seed=SEED), repeat_residual
        )

        residual_acceptance = tf.cast(residual_acceptance, tf.int64)
        repeat_count = tf.cast(repeat_count, dtype=tf.int64)

        return repeat_count + residual_acceptance

    return wrapper
    

# randomization for training sets
def create_training_dataset(file_names, batch_size, buffer_size, augmentations,
							use_sampling, probs, class_target_prob,  **kwargs):
	''' Create the training dataset from the TFRecords shard
	'''
	target_size = kwargs.get('target_size')
	channels = kwargs.get('channels')
	
	files = tf.data.Dataset.list_files(file_names, shuffle=True, seed=SEED)
	shards = files.shuffle(buffer_size=1000, seed=SEED)
    
	#for some reason causes majoy memory usage
	#dataset = shards.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), 
    #                            cycle_length=len(file_names), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	dataset = tf.data.TFRecordDataset(shards, compression_type='GZIP')
	dataset = dataset.shuffle(buffer_size=buffer_size, seed = SEED)
	dataset = dataset.map(parse_serialized_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # UNDER/OVERSAMPLING
	if use_sampling:
		print("using under/oversampling")
		z = oversampling_filter(probs, class_target_prob)
		dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensors(x).repeat(z(x)))    
		dataset = dataset.filter(undersampling_filter(probs, class_target_prob))
        
    # IMAGE PARSING: return (x,y)
	dataset = dataset.map(parse_image(target_size=target_size, 
								   channels = channels), 
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)	
    
    # AUGMENTATIONS: return (f(x), y)
	if augmentations:
		print("runtime initialized with augmentations")
		dataset = dataset.map(lambda x,y: (tf.numpy_function(apply_augmentation, [x], tf.float32),y))
	else:
		print("runtime initialized without augmentations")

    # BATCHING    
	dataset = dataset.batch(batch_size)
	print("autotune param", tf.data.experimental.AUTOTUNE)
	dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
	dataset = dataset.apply(tf.data.experimental.ignore_errors())
	return dataset


# Parsing TF fun for validation and testing
def validate(file_names, batch_size, **kwargs):
    target_size = kwargs.get('target_size')
    channels = kwargs.get('channels')

    files = tf.data.Dataset.list_files(file_names, shuffle=False, seed=SEED)
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(parse_serialized_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(parse_image(target_size=target_size, 
									  channels=channels), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    return dataset



def num_files(x):
	''''''
# label record iterator
	def parse_label(features):
	    labels = features['label']
	    y,_, counts = tf.unique_with_counts(labels)
	    return counts
	
	def count_labels(parser):
		cnt = parser.reduce(np.int64(0), lambda x, _: x + 1)
		print("total count is {}".format(cnt))


	data = tf.data.TFRecordDataset(x, compression_type='GZIP')
	data = data.map(parse_serialized_example, num_parallel_calls = tf.data.experimental.AUTOTUNE)
	labels = data.map(parse_label, num_parallel_calls = tf.data.experimental.AUTOTUNE)
	counts = count_labels(labels)
	print(" total number of {} found in the dataset".format(counts))
	return counts
	
