from __future__ import print_function
###############################################
#                   CIFAR                     #
# https://www.cs.toronto.edu/~kriz/cifar.html #
###############################################
import base64
import pickle
import os
import numpy as np
from matplotlib import image as mpimg
import tensorflow as tf

from .tfrecords_utils import *


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f, encoding='bytes')


class CIFAR10Converter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the CIFAR-10 dataset in `data_dir`"""
        self.data_dir = data_dir
        self.data = []
        # Train
        train_batches = []
        for i in range(5):
            b = os.path.join(self.data_dir, 'data_batch_%d' % (i + 1))
            if not os.path.isfile(b):
                print('Warning: Missing train batch', i + 1)
            else:
                train_batches.append(b)
        if len(train_batches):
            self.data.append(('train', train_batches))
        # Test
        test_batch = os.path.join(self.data_dir, 'test_batch')
        if not os.path.isfile(test_batch): 
            print('Warning: Missing test batch')
        else:
            self.data.append(('test', [test_batch]))
        # Labels
        self.label_names = unpickle(os.path.join(self.data_dir, 'batches.meta'))[b'label_names']

    def convert(self, tfrecords_path, save_image_in_records=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, data in self.data:
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            print('\nLoad', name)
            for i, item in enumerate(data):
                print('\rBatch %d/%d' % (i + 1, len(data)), end='')
                d = unpickle(item)
                for img, label in zip(d[b'data'], d[b'labels']):    
                    class_name = self.label_names[label]
                    img = np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0))
                    example = tf.train.Example(features=tf.train.Features(
                            feature={'image': bytes_feature([img.astype(np.uint8).tostring(order='C')]),
                                     'class': int64_feature([label]),
                                     'class_str': bytes_feature([base64.b64encode(class_name)])}))
                    writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
        print()
      
    
class CIFAR100Converter(CIFAR10Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the CIFAR-100 dataset in `data_dir`"""
        self.data_dir = data_dir
        self.data = []
        # Train
        for name in ['train', 'test']:
            batch = os.path.join(self.data_dir, name)
            if not os.path.isfile(batch): 
                print('Warning: Missing test batch')
            else:
                self.data.append((name, [batch]))
        # Labels
        self.label_names = unpickle(os.path.join(self.data_dir, 'meta'))[b'coarse_label_names']

    def convert(self, tfrecords_path, save_image_in_records=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, data in self.data:
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            print('\nLoad', name)
            for i, item in enumerate(data):
                print('\rBatch %d/%d' % (i + 1, len(data)), end='')
                d = unpickle(item)
                for img, label, coarse_label in zip(d[b'data'], d[b'fine_labels'], d[b'coarse_labels']):    
                    class_name = self.label_names[coarse_label]
                    img = np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0))
                    example = tf.train.Example(features=tf.train.Features(
                            feature={'image': bytes_feature([img.astype(np.uint8).tostring(order='C')]),
                                     'class': int64_feature([label]),
                                     'coarse_class': int64_feature([coarse_label]),
                                     'coarse_class_str': bytes_feature([base64.b64encode(class_name)])}))
                    writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
        print()
            
            
class CIFAR10Loader():    
    
    def __init__(self,
                 image_size=None,
                 verbose=False):
        """Init a Loader object."""
        self.image_size = image_size
        self.verbose = verbose
    
    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Basic features
        features = {'image' : tf.FixedLenFeature((), tf.string),
                    'class': tf.FixedLenFeature((), tf.int64),
                    'class_str': tf.FixedLenFeature((), tf.string),
                   }     
        parsed_features = tf.parse_single_example(example_proto, features)   
        image = decode_raw_image(parsed_features['image'], (32, 32, 3), image_size=self.image_size)
        parsed_features['image'] = tf.identity(image, name='image')
        parsed_features['class'] = tf.to_int32(parsed_features['class'])
        parsed_features['class_str'] = tf.decode_base64(parsed_features['class_str'])
        # Return
        if self.verbose: print_records(parsed_features)
        return parsed_features
    
    
class CIFAR100Loader(CIFAR10Loader): 
    
    def __init__(self, image_size=None, verbose=False):
        super(CIFAR100Loader, self).__init__(image_size=image_size, verbose=verbose)
    
    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Basic features
        features = {'image' : tf.FixedLenFeature((), tf.string),
                    'class': tf.FixedLenFeature((), tf.int64),
                    'coarse_class': tf.FixedLenFeature((), tf.int64),
                    'coarse_class_str': tf.FixedLenFeature((), tf.string),
                   }     
        parsed_features = tf.parse_single_example(example_proto, features)   
        image = decode_raw_image(parsed_features['image'], (32, 32, 3), image_size=self.image_size)
        parsed_features['image'] = tf.identity(image, name='image')
        parsed_features['class'] = tf.to_int32(parsed_features['class'])
        parsed_features['coarse_class'] = tf.to_int32(parsed_features['coarse_class'])
        parsed_features['coarse_class_str'] = tf.decode_base64(parsed_features['coarse_class_str'])
        # Return
        if self.verbose: print_records(parsed_features)
        return parsed_features