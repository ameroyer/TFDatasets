from __future__ import print_function
###########################################
#               SVHN dataset              #
# http://ufldl.stanford.edu/housenumbers/ #
###########################################
import os
import numpy as np
import scipy.io
from .tfrecords_utils import Converter, _bytes_feature, _floats_feature, _int64_feature 
import tensorflow as tf


def read_integer(bytel):
    return int('0x' + ''.join('{:02x}'.format(x) for x in bytel), 0)


class SVHNConverter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the SVHN dataset in `data_dir`"""
        print('Loading original SVHN data from', data_dir)
        self.train_data = os.path.join(data_dir, 'train_32x32.mat')
        if not os.path.isfile(self.train_data):             
            print('Warning: Missing training data')
            self.train_data = None
        self.val_data = os.path.join(data_dir, 'extra_32x32.mat')
        if not os.path.isfile(self.val_data):             
            print('Warning: Missing val (extra) data')
            self.val_data = None
        self.test_data = os.path.join(data_dir, 'test_32x32.mat')
        if not os.path.isfile(self.test_data):             
            print('Warning: Missing test data')
            self.test_data = None

    def convert(self, tfrecords_path):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, data in [['train', self.train_data], 
                           ['val', self.val_data],
                           ['test', self.test_data]]:    
            if data is None:          
                print('Warning: Missing %s data' % name)
                continue
            # Read images
            mat = scipy.io.loadmat(data)
            images, labels = mat['X'], mat['y']
            num_items = labels.shape[0]
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            for x in range(num_items):
                print('\rLoad %s: %d / %d' % (name, x + 1, num_items), end='')
                # load
                img = images[:, :, :, x]
                img = img.astype(np.uint8)
                class_id = int(labels[x, 0])
                class_id = 0 if class_id == 10 else class_id
                # write
                example = tf.train.Example(features=tf.train.Features(feature={
                    'class': _int64_feature([class_id]),
                    'image': _bytes_feature([img.tostring()])}))
                writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
            print()
            
            
class SVHNLoader():
    
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
        image = tf.reshape(image, (32, 32, 3))
        image = tf.image.convert_image_dtype(image, tf.float32)
        if self.image_resize is not None:
            image = tf.image.resize_images(image, (self.image_resize, self.image_resize))
        class_id = tf.to_int32(parsed_features['class'])
        return {'image': image, 'class': class_id}