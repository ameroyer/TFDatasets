from __future__ import print_function
########################################
#           Tiny ImageNet              #
# https://tiny-imagenet.herokuapp.com/ #
########################################
import base64
import os
import numpy as np
from matplotlib import image as mpimg
import tensorflow as tf

from .tfrecords_utils import *


"""Define features to be stored in the TFRecords"""
TinyImageNetFeatures = Features([
    ('image', FeatureType.BYTES, FeatureLength.FIXED, (), None),
    ('width', FeatureType.INT, FeatureLength.FIXED, (), tf.constant(-1, dtype=tf.int64)),
    ('height', FeatureType.INT, FeatureLength.FIXED, (), tf.constant(-1, dtype=tf.int64)),
    ('bounding_box', FeatureType.FLOAT, FeatureLength.FIXED, (4,), tf.constant([0., 0., 1., 1.])),
    ('class', FeatureType.INT, FeatureLength.FIXED, (), -1),
    ('class_str', FeatureType.BYTES, FeatureLength.FIXED, (), '')
                          ])

class TinyImageNetConverter(Converter):
    features = TinyImageNetFeatures

    def __init__(self, data_dir):
        """Initialize the object for the TinyImageNet dataset in `data_dir`"""
        self.data_dir = data_dir
        # Classes
        with open(os.path.join(data_dir, 'wnids.txt'), 'r') as f:
            self.synsets_to_ids = {synset: i for i, synset in enumerate(f.read().splitlines())}
        with open(os.path.join(data_dir, 'words.txt'), 'r') as f:
            self.synsets_to_labels = {items[0]: items[1] for line in f.read().splitlines() for items in [line.split('\t')]}
            self.synsets_to_labels = {k: self.synsets_to_labels[k] for k in self.synsets_to_ids}
        # Train
        self.data = []
        train_data = []
        for synset in os.listdir(os.path.join(self.data_dir, 'train')):
            lst = os.path.join(self.data_dir, 'train', synset, '%s_boxes.txt' % synset)
            if os.path.isfile(lst):
                with open(lst, 'r') as f:
                    train_data.extend([
                            (os.path.join('train', synset, 'images', items[0]), synset, np.array(list(map(int, items[1:]))))
                            for line in f.read().splitlines() for items in [line.split('\t')]])
            else:
                print('Warning: Missing %s train data' % synset)
        self.data.append(('train', train_data))
        # Val
        lst = os.path.join(self.data_dir, 'val', 'val_annotations.txt')
        if os.path.isfile(lst):
            with open(lst, 'r') as f:
                val_data = [(os.path.join('val', 'images', items[0]), items[1], np.array(list(map(int, items[2:]))))
                             for line in f.read().splitlines() for items in [line.split('\t')]]
                self.data.append(('val', val_data))
        else:
            print('Warning: Missing val data')
        # Test
        test_dir = os.path.join('test', 'images')
        full_test_dir = os.path.join(self.data_dir, test_dir)
        if os.path.exists(full_test_dir):
            test_data = [(os.path.join(test_dir, x),) for x in os.listdir(full_test_dir) if x.endswith('.JPEG')]
            self.data.append(('test', test_data))
        else:
            print('Warning: Missing test data')

    def convert(self,
                tfrecords_path,
                compression_type=None,
                save_image_in_records=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, data in self.data:
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = self.init_writer(writer_path, compression_type=compression_type)
            print('\nLoad', name)
            for i, item in enumerate(data):
                print('\rImage %d/%d' % (i + 1, len(data)), end='')
                # Image shape
                image_path = item[0]
                height, width = None, None
                if save_image_in_records or len(item) > 2:
                    img = mpimg.imread(os.path.join(self.data_dir, image_path))
                    height = [img.shape[0]]
                    width = [img.shape[1]]

                # Image
                img = (img.astype(np.uint8).tostring() if save_image_in_records else
                       base64.b64encode(image_path.encode('utf-8')))

                # Class
                class_id, class_name, bbox = None, None, None
                if len(item) > 1:
                    class_id = [self.synsets_to_ids[item[1]]]
                    class_name = self.synsets_to_labels[item[1]]
                    class_name = [base64.b64encode(class_name.encode('utf-8'))]

                    # Normalized Bounding box
                    if len(item) > 2:
                        bbox = np.array([item[2][1] / width[0], item[2][0] / height[0],
                                         item[2][3] / width[0], item[2][2] / height[0]], dtype=np.float32)
                        bbox = bbox.flatten()
                # Write
                writer.write(self.create_example_proto([img], height, width, bbox, class_id, class_name))
            writer.close()
            print('\nWrote %s in file %s (%.2fMB)' % (
                name, writer_path, os.path.getsize(writer_path) / 1e6))
        print()


class TinyImageNetLoader(Loader):
    features = TinyImageNetFeatures

    def __init__(self,
                 save_image_in_records=False,
                 image_dir='',
                 image_size=None,
                 verbose=False):
        """Init a Loader object.

        Args:
            `save_image_in_records` (bool): If True, the image was saved in the record, otherwise only the image path was.
            `image_dir` (str): If save_image_in_records is False, append this string to the image_path saved in the record.
            `image_size` (int): If given, resize the image to the given size
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
        parsed_features['class'] = tf.to_int32(parsed_features['class'])
        parsed_features['class_str'] = tf.decode_base64(parsed_features['class_str'])
        # Return
        del parsed_features['height']
        del parsed_features['width']
        if self.verbose: print_records(parsed_features)
        return parsed_features
