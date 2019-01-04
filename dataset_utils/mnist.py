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

"""Define features to be stored in the TFRecords"""
MNISTFeatures = Features([('class', FeatureType.INT, FeatureLength.FIXED, (),),
                          ('image', FeatureType.BYTES, FeatureLength.FIXED, (),),
                          ('id', FeatureType.INT, FeatureLength.FIXED, (),)])


class MNISTConverter(Converter):
    features = MNISTFeatures

    def __init__(self, data_dir):
        """Initialize the object for the MNIST dataset in `data_dir`"""
        print('Loading original MNIST data from', data_dir)
        self.data = []
        for name, key in [('train', 'train'), ('test', 't10k')]:
            images = os.path.join(data_dir, '%s-images.idx3-ubyte' % key)
            labels = os.path.join(data_dir, '%s-labels.idx1-ubyte' % key)
            if not os.path.isfile(images) or not os.path.isfile(labels):
                print('Warning: Missing %s data' % name)
            else:
                self.data.append((name, images, labels))

    def convert(self, tfrecords_path, compression_type=None, sort=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`
        If `sort` is True, the Example will be sorted by class in the final TFRecords.
        """
        for name, images, labels in self.data:
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
            writer = self.init_writer(writer_path, compression_type=compression_type)
            offset = 16
            num_pixels = num_rows * num_columns
            labels_order = np.argsort(blockLabels) if sort else range(num_items)
            for x, index in enumerate(labels_order):
                print('\rLoad %s: %d / %d' % (name, x + 1, num_items), end='')
                step = offset + index * num_pixels
                next_step = step + num_pixels
                img = np.array(block[step:next_step], dtype=np.uint8).reshape((num_rows, num_columns))
                class_id = blockLabels[index]
                writer.write(self.create_example_proto([class_id], [img.tostring()], [index]))
                step = next_step

            # End
            writer.close()
            print('\nWrote %s in file %s (%.2fMB)' % (
                name, writer_path, os.path.getsize(writer_path) / 1e6))
            print()


class MNISTLoader(Loader):
    features = MNISTFeatures
    shape = (28, 28, 1)

    def __init__(self, image_size=None, verbose=False):
        """Init a Loader object. Loaded images will be resized to size `image_size`."""
        self.image_size = image_size
        self.verbose = verbose

    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Parse
        parsed_features = self.raw_parsing_fn(example_proto)
        # Reshape
        parsed_features['image'] = decode_raw_image(parsed_features['image'], self.shape, image_size=self.image_size)
        parsed_features['image'] = tf.identity(parsed_features['image'], name='image')
        parsed_features['class'] = tf.to_int32(parsed_features['class'], name='class_label')
        parsed_features['id'] = tf.to_int32(parsed_features['id'], name='index')
        # Return
        if self.verbose: print_records(parsed_features)
        return parsed_features
