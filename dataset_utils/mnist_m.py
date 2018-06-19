from __future__ import print_function
##############################
#      MNIST-M dataset       #
# http://yaroslav.ganin.net/ #
##############################
import os
import numpy as np
from matplotlib import image as mpimg
from .tfrecords_utils import Converter, _bytes_feature, _floats_feature, _int64_feature 
import tensorflow as tf


class MNISTMConverter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the MNIST-M dataset in `data_dir`"""
        print('Loading original MNIST-M data from', data_dir)
        self.data_dir = data_dir
        train_split = os.path.join(data_dir, 'mnist_m_train_labels.txt')
        if not os.path.isfile(train_split):             
            print('Warning: Missing training data')
            self.train_images, self.train_labels = None, None
        else:
            with open(train_split, 'r') as f:
                self.train_images, self.train_labels = zip(*[line.split() for line in f.read().splitlines()])
                self.train_labels = list(map(int, self.train_labels))
        test_split = os.path.join(data_dir, 'mnist_m_test_labels.txt')
        if not os.path.isfile(test_split):             
            print('Warning: Missing test data')
            self.test_images, self.test_labels = None, None
        else:
            with open(test_split, 'r') as f:
                self.test_images, self.test_labels = zip(*[line.split() for line in f.read().splitlines()])
                self.test_labels = list(map(int, self.test_labels))

    def convert(self, tfrecords_path, sort=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, images, labels in [['train', self.train_images, self.train_labels], 
                                     ['test', self.test_images, self.test_labels]]: 
            if images is None or labels is None:   
                continue
            image_dir = os.path.join(self.data_dir, 'mnist_m_%s' % name)
            if not os.path.exists(image_dir):
                print('Warning: Missing %s image directory' % name)
                continue
            # Sort labels
            num_items = len(labels)
            if sort:
                labels_order = np.argsort(labels, axis=0)
            else:
                labels_order = range(num_items)
            # Read images
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            for x, index in enumerate(labels_order):
                print('\rLoad %s: %d / %d' % (name, x + 1, num_items), end='')
                # load
                img = np.ceil(255. * mpimg.imread(os.path.join(image_dir, images[index])))
                img = img.astype(np.uint8)
                class_id = labels[index]
                # write
                example = tf.train.Example(features=tf.train.Features(feature={
                    'class': _int64_feature([class_id]),
                    'image': _bytes_feature([img.tostring()]),
                    'id': _int64_feature([index])}))
                writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
            print()
            
            
class MNISTMLoader():
    
    def __init__(self, resize=None):
        """Init a Loader object. Loaded images will be resized to size `resize`."""
        self.image_resize = resize
    
    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Basic features
        features = {'class': tf.FixedLenFeature((), tf.int64),
                    'image': tf.FixedLenFeature((), tf.string),
                    'id': tf.FixedLenFeature((), tf.int64)}      
        parsed_features = tf.parse_single_example(example_proto, features)  
        image = tf.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.reshape(image, (32, 32, 3))
        image = tf.image.convert_image_dtype(image, tf.float32)
        if self.image_resize is not None:
            image = tf.image.resize_images(image, (self.image_resize, self.image_resize))
        class_id = tf.to_int32(parsed_features['class'])
        index = tf.to_int32(parsed_features['id'])
        return {'image': image, 'class': class_id, 'id': index}