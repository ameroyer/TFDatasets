from __future__ import print_function
###########################################
#               SVHN dataset              #
# http://ufldl.stanford.edu/housenumbers/ #
###########################################
import os
import numpy as np
import scipy.io
import tensorflow as tf

from .tfrecords_utils import *


class SVHNConverter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the SVHN dataset in `data_dir`"""
        print('Loading original SVHN data from', data_dir)
        self.data = []
        for name, key in [('train', 'train'), ('val', 'extra'), ('test', 'test')]:
            data = os.path.join(data_dir, '%s_32x32.mat' % key)
            if not os.path.isfile(data):             
                print('Warning: Missing %s data' % name)
            else:
                self.data.append((name, data))

    def convert(self, tfrecords_path, sort=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, data in self.data:
            # Load
            mat = scipy.io.loadmat(data)
            images, labels = mat['X'], mat['y']
            num_items = labels.shape[0]
            # Write
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            labels_order = np.argsort(labels, axis=0) if sort else range(num_items)
            for x, index in enumerate(labels_order):
                print('\rLoad %s: %d / %d' % (name, x + 1, num_items), end='')
                img = images[:, :, :, index]
                img = img.astype(np.uint8)
                class_id = int(labels[index, 0])
                class_id = 0 if class_id == 10 else class_id
                example = tf.train.Example(features=tf.train.Features(feature={
                    'class': int64_feature([class_id]),
                    'image': bytes_feature([img.tostring()]),
                    'id': int64_feature([index])}))
                writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
            print()
            
            
class SVHNLoader():
    
    def __init__(self, image_size=None, verbose=False):
        """Init a Loader object. Loaded images will be resized to size `resize`."""
        self.image_size = image_size
        self.verbose = verbose
    
    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Basic features
        features = {'class': tf.FixedLenFeature((), tf.int64),
                    'image': tf.FixedLenFeature((), tf.string),
                    'id': tf.FixedLenFeature((), tf.int64)}      
        parsed_features = tf.parse_single_example(example_proto, features) 
        image = decode_raw_image(parsed_features['image'], (32, 32, 3), image_size=self.image_size)
        image = tf.identity(image, name='image')
        class_id = tf.to_int32(parsed_features['class'], name='class_label')
        index = tf.to_int32(parsed_features['id'], name='index')
        # Return
        records_dict = {'image': image, 'class': class_id, 'id': index}
        if self.verbose: print_records(records_dict)
        return records_dict