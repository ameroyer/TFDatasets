from __future__ import print_function
################################
#        VisDA dataset         #
# http://ai.bu.edu/visda-2017/ #
################################
import base64
import os
import numpy as np
from matplotlib import image as mpimg
import tensorflow as tf

from .tfrecords_utils import *


class VisdaClassificationConverter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the VisDA dataset in `data_dir`"""
        self.data_dir = data_dir
        self.data = []
        for name, key in [('train', 'train'), ('val', 'validation'), ('test', 'test')]:
            lst = os.path.join(data_dir, key, 'image_list.txt')
            if os.path.exists(lst):
                with open(lst, 'r') as f:
                    data = [(os.path.join(key, items[0]), int(items[1])) if len(items) > 1 else (os.path.join(name, line),)
                            for line in f.read().splitlines() for items in [line.split()]]
                self.data.append((name, data))
            else:
                print('Warning: No %s data found' % name)

    def convert(self,
                tfrecords_path, 
                save_image_in_records=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, split in self.data:    
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            # For each dir
            print('\nLoad', name)
            for i, aux in enumerate(split):
                print('\rImage %d/%d' % (i + 1, len(split)), end='')
                feature = {}
                # Image
                image_path = aux[0]
                if save_image_in_records:
                    img = mpimg.imread(os.path.join(self.data_dir, image_path))
                    if name == 'train': # synthetic
                        img = img * 255.
                    feature['image'] = bytes_feature([img.astype(np.uint8).tostring()])
                    feature['width'] = int64_feature([img.shape[0]])
                    feature['height'] = int64_feature([img.shape[1]])
                else:
                    feature['image'] = bytes_feature([base64.b64encode(image_path.encode('utf-8'))])
                # Class
                if len(aux) > 1:
                    feature['class'] = int64_feature([aux[1]])
                # Write
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
        print()
            
            
class VisdaClassificationLoader():
    classes_names = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 
                     'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
    
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
        self.image_size = image_resize
        self.verbose = verbose
    
    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Basic features
        features = {'image' : tf.FixedLenFeature((), tf.string),
                    'class': tf.FixedLenFeature((), tf.int64, default_value=tf.constant(-1, dtype=tf.int64)),
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
        # Class
        parsed_features['class'] = tf.to_int32(parsed_features['class'], name='class')
        # Return
        del parsed_features['height']
        del parsed_features['width']
        if self.verbose: print_records(parsed_features)
        return parsed_features