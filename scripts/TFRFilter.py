# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:36:58 2019

@author: Stephan
"""

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import random
import os


def load_dataset(filename, num_calls = tf.data.experimental.AUTOTUNE, compression=None):
    ''' Load a TFRecord dataset 
    
    Yields a dataset from a filename
    
    Args:
        filename: the target path containing the path and the filename to the tensor
        
    Returns 
        A mapped dataset specified by parse_tfrecord
    '''
    
    def parse_tfrecord(example_proto):
        ''' Parses a single example from the protobuffer
        
        Args:
            example_proto: a serialized example
        
        returns:
            A mapping (dictionary) to the features specified in featuresDict
        '''
        featuresDict = {
            'B2': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # B
            'B3': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # G
            'B4': tf.io.FixedLenFeature([257, 257], dtype=tf.float32),  # R
            #'AVE': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # Elevation
            #'NDVI': tf.io.FixedLenFeature([257, 257], dtype=tf.float32), # vegetation index
            'class': tf.io.FixedLenFeature([1], dtype=tf.float32)
        }
        return tf.io.parse_single_example(example_proto, featuresDict)

    dataset = tf.data.TFRecordDataset(filename, compression_type=compression)
    return(dataset.map(parse_tfrecord, num_parallel_calls=num_calls))



def wrapped_feature(feature):
    ''' Creates a feature to write to TFRecord
    
    Args:
        feature: a tuple with the feature and the index. The index is added to visualize the sample randomness later on
    
    returns: 
        a feature that is writable to a TFRecord
    '''
    #Quick little fix for multiprocessing...
    i = feature[0]
    feature = feature[1]
    def create_feature(feature):
        ''' creates an example
        
        Args:
            feature the feature to be processed
        '''
        # Create the Example
        example = tf.train.Example(features=tf.train.Features(feature={
            'B2': tf.train.Feature(float_list=tf.train.FloatList(value=feature['B2'].numpy().flatten())), 
            'B3': tf.train.Feature(float_list=tf.train.FloatList(value=feature['B3'].numpy().flatten())),
            'B4': tf.train.Feature(float_list=tf.train.FloatList(value=feature['B4'].numpy().flatten())),
            'class': tf.train.Feature(float_list=tf.train.FloatList(value=feature['class'].numpy())),
            'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
        }))
        return example
    return create_feature(feature)


class suppressed_iterator:
    def __init__(self, wrapped_iter, skipped_exc = tf.errors.InvalidArgumentError):
        self.wrapped_iter = wrapped_iter
        self.skipped_exc  = skipped_exc

    def __next__(self):
        while True:
            try:
                return next(self.wrapped_iter)
            except StopIteration:
                raise
            except self.skipped_exc:
                pass

class suppressed_generator:
    def __init__(self, wrapped_obj, skipped_exc = tf.errors.InvalidArgumentError):
        self.wrapped_obj = wrapped_obj
        self.skipped_exc = skipped_exc

    def __iter__(self):
        return suppressed_iterator(iter(self.wrapped_obj), self.skipped_exc)


class TFRecordGenerator:
    ''' Class for generating tfrecord: from https://colab.research.google.com/
    github/christianmerkwirth/colabs/blob/master/
    Understanding_Randomization_in_TF_Datasets.ipynb#scrollTo=pS0ihDFTd1uI
    
    creates a sharded set of TRFecords from a single TFRecord
    
    '''
    def __init__(self, num_shards=10):
        self.num_shards = num_shards

    def _pick_output_shard(self):
        return random.randint(0, self.num_shards-1)

    def generate_records(self, dataset, tfrecord_file_name):
        ''' Creates a set of TFRecords from a TF dataset
        
        Args:
            dataset: the TFRecord dataset to be split into shards
            tfrecord_file_name: the path to write the TFRecords to
            
        returns:
            void: writes the new TFRecord shards to the path specified in tfrecord_file_name
                  if only a filename is specified, then they will be in the directory from where the file is executed.
        
        '''
        writers = []
        options = tf.io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
        for i in range(self.num_shards):
            writers.append(
                    tf.io.TFRecordWriter("{}-{}-of-{}.gz".format(tfrecord_file_name, i, self.num_shards), options))  
        counter = 0    
        for x in enumerate(suppressed_generator(dataset)):
            example = wrapped_feature(x)
            writers[self._pick_output_shard()].write(example.SerializeToString())
            counter += 1
            
            if x[0] % 100 == 0:
                print("Currently at index {}".format(x[0]))

        # Close all files
        for w in writers:
            w.close()
        print("%d records writen" % counter)
    
    def generate_records_mp(self, dataset, tfrecord_file_name):
        ''' Creates a set of TFRecords from a TF dataset in multiprocessing style
        
        Args:
            dataset: the TFRecord dataset to be split into shards
            tfrecord_file_name: the path to write the TFRecords to
            
        returns:
            void: writes the new TFRecord shards to the path specified in tfrecord_file_name
                  if only a filename is specified, then they will be in the directory from where the file is executed.
        
        warning: the entire single TFRecord is first put into a list, i.e. read into memory. 
        This can lead to memory errors for large files.
        
        '''
        writers = []
        options = tf.io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
        
        for i in range(self.num_shards):
            writers.append(
                    tf.io.TFRecordWriter("{}-{}-of-{}.gz".format(tfrecord_file_name, i, self.num_shards), options))  
                
        counter = 0    
        print("converting TFRecord to list")
        parser = enumerate(list(suppressed_generator(dataset)))
        
        print("starting multiprocessing threads")
        with Pool(10) as p:  
            for result in p.map(wrapped_feature, parser):
                writers[self._pick_output_shard()].write(result.SerializeToString())
                counter +=1 
        
        print("multiprocessing completed succesfully, files are in {}".format(tfrecord_file_name))
        # Close all files
        for w in writers:
            w.close()
        print("%d records writen" % counter)
        

    
if __name__ == '__main__':
    data_path = os.path.join('..', 'data', 'raw')
    out_path = os.path.join('..', 'data')
    
    # training and testing locations relative to this scripts location
    train_file = os.path.join(data_path, 'training_WaterEdges.gz')
    out_files_train = os.path.join(out_path, 'training_WaterEdges')
    
    test_file = os.path.join(data_path, 'testing_WaterEdges.gz')
    out_files_test = os.path.join(out_path, 'test_WaterEdges')


    
    # split dataset into shards
    t = TFRecordGenerator(num_shards = 7)
    
    for infile, outfile in [(train_file, out_files_train), (test_file, out_files_test)]:
        # load dataset
        print("loading dataset located in {}".format(infile))
        dataset = load_dataset(infile, compression='GZIP')
        t.generate_records_mp(dataset, outfile)
    