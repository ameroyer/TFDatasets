from __future__ import print_function
#####################################
#           MNIST dataset           #
# http://yann.lecun.com/exdb/mnist/ #
#####################################
import codecs
import os
import numpy as np
from .tfrecords_utils import Converter, _bytes_feature, _floats_feature, _int64_feature 
import tensorflow as tf


def read_integer(bytel):
    return int('0x' + ''.join('{:02x}'.format(x) for x in bytel), 0)


class MNISTConverter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the MNIST dataset in `data_dir`"""
        print('Loading original MNIST data from', data_dir)
        self.train_images = os.path.join(data_dir, 'train-images.idx3-ubyte')
        self.train_labels = os.path.join(data_dir, 'train-labels.idx1-ubyte')
        if not os.path.isfile(self.train_images) or not os.path.isfile(self.train_labels):             
            print('Warning: Missing training data')
            self.train_images = None
            self.train_labels = None
        self.test_images = os.path.join(data_dir, 't10k-images.idx3-ubyte')
        self.test_labels = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
        if not os.path.isfile(self.test_images) or not os.path.isfile(self.test_labels):             
            print('Warning: Missing test data')
            self.test_images = None
            self.test_labels = None

    def convert(self, tfrecords_path):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, images, labels in [['train', self.train_images, self.train_labels], 
                                     ['test', self.test_images, self.test_labels]]:    
            if images is None or labels is None:          
                print('Warning: Missing %s data' % name)
                continue
            # Read images
            with codecs.open(images, 'r', 'latin-1') as f:
                block = list(bytearray(f.read(), 'latin-1'))
                assert(not(block[0] | block[1]))
            num_items = read_integer(block[4:8])
            num_rows = read_integer(block[8:12])
            num_columns = read_integer(block[12:16])
            # Read labels
            with codecs.open(labels, 'r', 'latin-1') as f:
                blockLabels = list(bytearray(f.read(), 'latin-1'))[8:]
            # Parse
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            step = 16
            for x in range(num_items):
                print('\rLoad %s: %d / %d' % (name, x + 1, num_items), end='')
                # load
                next_step = step + num_rows * num_columns
                img = np.array(block[step : next_step]).reshape((num_rows, num_columns))
                img = img.astype(np.uint8)
                class_id = blockLabels[x]
                step = next_step
                # write
                example = tf.train.Example(features=tf.train.Features(feature={
                    'class': _int64_feature([class_id]),
                    'image': _bytes_feature([img.tostring()])}))
                writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
            print()
            
            
class MNISTLoader():
    
    def __init__(self, resize=None):
        """Init a Loader object. Loaded images will be resized to size `resize`."""
        self.image_resize = resize
    
    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Basic features
        features = {'class': tf.FixedLenFeature((), tf.int64),
                    'image': tf.FixedLenFeature((), tf.string)}      
        parsed_features = tf.parse_single_example(example_proto, features)  
        image = tf.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.reshape(image, (28, 28, 1))
        image = tf.image.convert_image_dtype(image, tf.float32)
        if self.image_resize is not None:
            image = tf.image.resize_images(image, (self.image_resize, self.image_resize))
        class_id = tf.to_int32(parsed_features['class'])
        return {'image': image, 'class': class_id}