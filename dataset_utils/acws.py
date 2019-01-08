from __future__ import print_function
######################################################
#                  ACWS dataset                      #
#https://data.vision.ee.ethz.ch/cvl/lbossard/accv12/ #
######################################################
import base64
import os
import numpy as np
from matplotlib import image as mpimg
import tensorflow as tf

from .tfrecords_utils import *

"""Define features to be stored in the TFRecords"""
ACWSFeatures = Features([
    ('image', FeatureType.BYTES, FeatureLength.FIXED, (), None),
    ('class', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('width', FeatureType.INT, FeatureLength.FIXED, (), tf.constant(-1, dtype=tf.int64)),
    ('height', FeatureType.INT, FeatureLength.FIXED, (), tf.constant(-1, dtype=tf.int64)),
                        ])

class ACWSConverter(Converter):
    features = ACWSFeatures

    def __init__(self, data_dir):
        """Initialize the object for the ACWS dataset in `data_dir`"""
        self.image_dir = os.path.join(data_dir, 'images')
        self.data = []
        for name in ['train', 'test']:
            lst = os.path.join(data_dir, '%s.txt' % name)
            if not os.path.isfile(lst):
                print('Warning: Missing %s data split' % name)
            else:
                with open(lst, 'r') as f:
                    split = f.read().splitlines()
                    self.data.append((name, split))

    def convert(self,
                tfrecords_path,
                compression_type=None,
                save_image_in_records=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, split in self.data:
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = self.init_writer(writer_path, compression_type=compression_type)
            print('\nLoad', name)
            for i, path in enumerate(split):
                print('\rImage %d/%d' % (i + 1, len(split)), end='')
                # Image
                image_path = '%s.jpg' % path
                if save_image_in_records:
                    img = mpimg.imread(os.path.join(self.image_dir, image_path))
                    height, width = img.shape[:2]
                    img = img.astype(np.uint8).tostring()
                else:
                    img = base64.b64encode(image_path.encode('utf-8'))
                # Class
                class_id = int(path.split('/', 1)[0])
                # Write
                writer.write(self.create_example_proto([img],
                                                       [class_id],
                                                       [height] if save_image_in_records else None,
                                                       [width] if save_image_in_records else None))
            writer.close()
            print('\nWrote %s in file %s (%.2fMB)' % (
                name, writer_path, os.path.getsize(writer_path) / 1e6))
        print()


class ACWSLoader(Loader):
    features = ACWSFeatures
    classes_names = ['blouses', 'cloak', 'coat', 'jacket', 'long dress', 'polo shirt, sport shirt', 'robe',
                     'shirt', 'short dress', 'suit, suit of clothes', 'sweater', 'jersey, T-shirt, tee shirt',
                     'undergarment, upper body', 'uniform', 'vest, waistcoat']

    def __init__(self,
                 save_image_in_records=False,
                 image_dir='',
                 image_size=None,
                 verbose=False):
        """Init a Loader object.

        Args:
            `save_image_in_records` (bool): If True, the image was saved in the record, otherwise only the image path was.
            `data_dir` (str): If save_image_in_records is False, append this string to the image_path saved in the record.
            `resize` (int): If given, resize the image to the given size
        """
        self.save_image_in_records = save_image_in_records
        self.image_dir = image_dir
        self.image_size = image_size
        self.verbose = verbose

    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Parse
        parsed_features = self.raw_parsing_fn(example_proto)
        # Reshape
        if self.save_image_in_records:
            shape = tf.stack([parsed_features['width'], parsed_features['height'], 3], axis=0)
            image = decode_raw_image(parsed_features['image'], shape, image_size=self.image_size)
        else:
            filename = tf.decode_base64(parsed_features['image'])
            parsed_features['image_path'] = tf.identity(filename, name='image_path')
            image = decode_relative_image(filename, self.image_dir, image_size=self.image_size)
        parsed_features['image'] = tf.identity(image, name='image')
        parsed_features['class'] = tf.to_int32(parsed_features['class'], name='class')
        # Return
        del parsed_features['height']
        del parsed_features['width']
        if self.verbose: print_records(parsed_features)
        return parsed_features
