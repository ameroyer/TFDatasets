from __future__ import print_function
###############################################
#                   CIFAR                     #
# https://www.cs.toronto.edu/~kriz/cifar.html #
###############################################
import base64
import pickle
import os
import numpy as np
import tensorflow as tf

from .tfrecords_utils import *


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f, encoding='bytes')


"""Define features to be stored in the TFRecords"""
CIFAR10Features = Features([('image', FeatureType.BYTES, FeatureLength.FIXED, (), None),
                            ('class', FeatureType.INT, FeatureLength.FIXED, (), None),
                            ('class_str', FeatureType.BYTES, FeatureLength.FIXED, (), None)])

CIFAR100Features = Features([('image', FeatureType.BYTES, FeatureLength.FIXED, (), None),
                             ('class', FeatureType.INT, FeatureLength.FIXED, (), None),
                             ('coarse_class', FeatureType.INT, FeatureLength.FIXED, (), None),
                             ('coarse_class_str', FeatureType.BYTES, FeatureLength.FIXED, (), None)])


class CIFAR10Converter(Converter):
    features = CIFAR10Features

    def __init__(self, data_dir):
        """Initialize the object for the CIFAR-10 dataset in `data_dir`"""
        self.data = []
        # Train
        train_batches = []
        for i in range(5):
            b = os.path.join(data_dir, 'data_batch_%d' % (i + 1))
            if not os.path.isfile(b):
                print('Warning: Missing train batch', i + 1)
            else:
                train_batches.append(b)
        if len(train_batches):
            self.data.append(('train', train_batches))
        # Test
        test_batch = os.path.join(data_dir, 'test_batch')
        if not os.path.isfile(test_batch):
            print('Warning: Missing test batch')
        else:
            self.data.append(('test', [test_batch]))
        # Labels
        self.label_names = unpickle(os.path.join(data_dir, 'batches.meta'))[b'label_names']

    def convert(self, tfrecords_path, compression_type=None):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, data in self.data:
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = self.init_writer(writer_path, compression_type=compression_type)
            print('\nLoad', name)
            for i, item in enumerate(data):
                print('\rBatch %d/%d' % (i + 1, len(data)), end='')
                d = unpickle(item)
                for img, label in zip(d[b'data'], d[b'labels']):
                    class_name = self.label_names[label]
                    img = np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0))
                    writer.write(self.create_example_proto([img.astype(np.uint8).tostring(order='C')],
                                                           [label],
                                                           [base64.b64encode(class_name)]))
            writer.close()
            print('\nWrote %s in file %s (%.2fMB)' % (
                name, writer_path, os.path.getsize(writer_path) / 1e6))
        print()


class CIFAR100Converter(CIFAR10Converter):
    features = CIFAR100Features

    def __init__(self, data_dir):
        """Initialize the object for the CIFAR-100 dataset in `data_dir`"""
        self.data = []
        # Train
        for name in ['train', 'test']:
            batch = os.path.join(data_dir, name)
            if not os.path.isfile(batch):
                print('Warning: Missing test batch')
            else:
                self.data.append((name, [batch]))
        # Labels
        self.label_names = unpickle(os.path.join(data_dir, 'meta'))[b'coarse_label_names']

    def convert(self, tfrecords_path, compression_type=None):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, data in self.data:
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = self.init_writer(writer_path, compression_type=compression_type)
            print('\nLoad', name)
            for i, item in enumerate(data):
                print('\rBatch %d/%d' % (i + 1, len(data)), end='')
                d = unpickle(item)
                for img, label, coarse_label in zip(d[b'data'], d[b'fine_labels'], d[b'coarse_labels']):
                    class_name = self.label_names[coarse_label]
                    img = np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0))
                    writer.write(self.create_example_proto([img.astype(np.uint8).tostring(order='C')],
                                                           [label], [coarse_label],
                                                           [base64.b64encode(class_name)]))
            writer.close()
            print('\nWrote %s in file %s (%.2fMB)' % (
                name, writer_path, os.path.getsize(writer_path) / 1e6))
        print()


class CIFAR10Loader(Loader):
    features = CIFAR10Features

    def __init__(self,
                 image_size=None,
                 verbose=False):
        """Init a Loader object."""
        self.image_size = image_size
        self.verbose = verbose

    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Parse
        parsed_features = self.raw_parsing_fn(example_proto)
        # Reshape
        parsed_features['image'] = decode_raw_image(parsed_features['image'],
                                                    (32, 32, 3), image_size=self.image_size)
        parsed_features['image'] = tf.identity(parsed_features['image'], name='image')
        parsed_features['class'] = tf.to_int32(parsed_features['class'])
        parsed_features['class_str'] = tf.decode_base64(parsed_features['class_str'])
        # Return
        if self.verbose: print_records(parsed_features)
        return parsed_features


class CIFAR100Loader(CIFAR10Loader):
    features = CIFAR100Features

    def __init__(self, image_size=None, verbose=False):
        super(CIFAR100Loader, self).__init__(image_size=image_size, verbose=verbose)

    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Parse
        parsed_features = self.raw_parsing_fn(example_proto)
        # Reshape
        parsed_features['image'] = decode_raw_image(parsed_features['image'],
                                                    (32, 32, 3), image_size=self.image_size)
        parsed_features['image'] = tf.identity(parsed_features['image'], name='image')
        parsed_features['class'] = tf.to_int32(parsed_features['class'])
        parsed_features['coarse_class'] = tf.to_int32(parsed_features['coarse_class'])
        parsed_features['coarse_class_str'] = tf.decode_base64(parsed_features['coarse_class_str'])
        # Return
        if self.verbose: print_records(parsed_features)
        return parsed_features
