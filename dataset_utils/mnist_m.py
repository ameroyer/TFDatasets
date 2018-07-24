from __future__ import print_function
##############################
#      MNIST-M dataset       #
# http://yaroslav.ganin.net/ #
##############################
import os
import numpy as np
from matplotlib import image as mpimg
import tensorflow as tf

from .tfrecords_utils import * 


class MNISTMConverter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the MNIST-M dataset in `data_dir`"""
        print('Loading original MNIST-M data from', data_dir)
        self.data_dir = data_dir
        self.data = []
        for name in ['train', 'test']:
            split = os.path.join(data_dir, 'mnist_m_%s_labels.txt' % name)
            image_dir = os.path.join(self.data_dir, 'mnist_m_%s' % name)
            if not os.path.isfile(split):             
                print('Warning: Missing %s data' % name)
            elif not os.path.exists(image_dir):
                print('Warning: Missing %s image directory' % name)
            else:
                with open(split, 'r') as f:
                    images, labels = zip(*[line.split() for line in f.read().splitlines()])
                    labels = list(map(int, labels))
                    self.data.append((name, image_dir, images, labels))

    def convert(self, tfrecords_path, sort=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, image_dir, images, labels in self.data:
            num_items = len(labels)
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            labels_order = np.argsort(labels, axis=0) if sort else range(num_items)
            for x, index in enumerate(labels_order):
                print('\rLoad %s: %d / %d' % (name, x + 1, num_items), end='')
                img = np.ceil(255. * mpimg.imread(os.path.join(image_dir, images[index])))
                img = img.astype(np.uint8)
                class_id = labels[index]
                example = tf.train.Example(features=tf.train.Features(feature={
                    'class': int64_feature([class_id]),
                    'image': bytes_feature([img.tostring()]),
                    'id': int64_feature([index])}))
                writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
            print()
            
            
class MNISTMLoader():
    
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