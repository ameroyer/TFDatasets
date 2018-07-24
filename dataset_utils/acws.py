from __future__ import print_function
#####################################################################
#                           ACWS dataset                            #
# http://www.vision.ee.ethz.ch/~lbossard/projects/accv12/index.html #
#####################################################################
import base64
import os
import numpy as np
from matplotlib import image as mpimg
import tensorflow as tf

from .tfrecords_utils import *


class ACWSConverter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the ACWSdataset in `data_dir`"""
        self.image_dir = os.path.join(data_dir, 'images')
        assert os.path.exists(self.image_dir)
        self.data = []
        for name in ['train', 'test']:
            lst = os.path.join(data_dir, '%s.txt' % name)
            if not os.path.isfile(lst):             
                print('Warning: Missing %s data split' % name)
            else:
                with open(lst, 'r') as f:
                    split = f.read().splitlines()
                    self.data.append((name, split))

    def convert(self, tfrecords_path, save_image_in_records=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, split in self.data:
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            print('\nLoad', name)
            for i, path in enumerate(split):
                print('\rImage %d/%d' % (i + 1, len(split)), end='')
                feature = {}
                # Image
                image_path = '%s.jpg' % path
                if save_image_in_records:
                    img = mpimg.imread(os.path.join(self.image_dir, image_path))
                    height, width = img.shape[:2]
                    feature['image'] = bytes_feature([img.astype(np.uint8).tostring()])
                    feature['width'] = int64_feature([img.shape[0]])
                    feature['height'] = int64_feature([img.shape[1]])
                else:
                    feature['image'] = bytes_feature([base64.b64encode(image_path.encode('utf-8'))])
                # Class
                class_id = int(path.split('/', 1)[0])
                feature['class'] = int64_feature([class_id])
                # Write
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
        print()
            
            
class ACWSLoader():
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
        # Basic features
        features = {'image' : tf.FixedLenFeature((), tf.string),
                    'class': tf.FixedLenFeature((), tf.int64),
                    'height': tf.FixedLenFeature((), tf.int64, default_value=tf.constant(-1, dtype=tf.int64)),
                    'width': tf.FixedLenFeature((), tf.int64, default_value=tf.constant(-1, dtype=tf.int64))
                   }     
        parsed_features = tf.parse_single_example(example_proto, features)   
        # Load image
        if self.save_image_in_records: 
            shape = tf.stack([parsed_features['width'], parsed_features['height'], 3], axis=0)
            image = decode_raw_image(parsed_features['image'], shape, image_size=self.image_size)
        else:
            filename = tf.decode_base64(parsed_features['image'])
            parsed_features['image_path'] = tf.identity(filename, name='image_path')
            image = decode_relative_image(filename, self.image_dir, image_size=self.image_size)
        parsed_features['image'] = tf.identity(image, name='image')
        parsed_features['class'] = tf.to_int32(parsed_features['class'])
        # Return
        del parsed_features['height']
        del parsed_features['width']
        if self.verbose: print_records(parsed_features)
        return parsed_features