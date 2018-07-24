from __future__ import print_function
#####################################
#           MNIST dataset           #
# http://yann.lecun.com/exdb/mnist/ #
#####################################
import codecs
import os
import numpy as np
import tensorflow as tf
from .tfrecords_utils import * 


def read_integer(bytel):
    return int('0x' + ''.join('{:02x}'.format(x) for x in bytel), 0)


class MNISTConverter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the MNIST dataset in `data_dir`"""
        print('Loading original MNIST data from', data_dir)
        self.data = []
        for name, key in [('train', 'train'), ('test', 't10k')]:
            images = os.path.join(data_dir, '%s-images.idx3-ubyte' % key)
            labels = os.path.join(data_dir, '%s-labels.idx1-ubyte' % key)
            if not os.path.isfile(images) or not os.path.isfile(labels):             
                print('Warning: Missing training data')
            else:
                self.data.append((name, images, labels))

    def convert(self, tfrecords_path, sort=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`
        If `sort` is True, the Example will be sorted by class in the final TFRecords.
        """
        for name, images, labels in self.data:    
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
            # Write
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            offset = 16
            num_pixels = num_rows * num_columns
            labels_order = np.argsort(blockLabels) if sort else range(num_items)
            for x, index in enumerate(labels_order):
                print('\rLoad %s: %d / %d' % (name, x + 1, num_items), end='')
                step = offset + index * num_pixels
                next_step = step + num_pixels
                img = np.array(block[step:next_step]).reshape((num_rows, num_columns))
                img = img.astype(np.uint8)
                class_id = blockLabels[index]
                step = next_step
                example = tf.train.Example(features=tf.train.Features(feature={
                    'class': int64_feature([class_id]),
                    'image': bytes_feature([img.tostring()]),
                    'id': int64_feature([index])}))                
                writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
            print()
            
            
class MNISTLoader():
    
    def __init__(self, resize=None, verbose=False):
        """Init a Loader object. Loaded images will be resized to size `resize`."""
        self.image_resize = resize
        self.verbose = verbose
    
    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Basic features
        features = {'class': tf.FixedLenFeature((), tf.int64),
                    'image': tf.FixedLenFeature((), tf.string),
                    'id': tf.FixedLenFeature((), tf.int64)}      
        parsed_features = tf.parse_single_example(example_proto, features)  
        # Parse
        image = decode_raw_image(parsed_features['image'], (28, 28, 1), image_size=None)
        image = tf.identity(image, name='image')
        class_id = tf.to_int32(parsed_features['class'], name='class_label')
        index = tf.to_int32(parsed_features['id'], name='index')
        # Return
        records_dict = {'image': image, 'class': class_id, 'id': index}
        if self.verbose: print_records(records_dict)
        return records_dict