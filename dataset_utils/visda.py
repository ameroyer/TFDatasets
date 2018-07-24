from __future__ import print_function
################################
#        VisDA dataset         #
# http://ai.bu.edu/visda-2017/ #
################################
import base64
import csv
import os
import numpy as np
from matplotlib import image as mpimg
from .tfrecords_utils import *
import tensorflow as tf


class VisdaClassificationConverter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the VisDA dataset in `data_dir`"""
        self.data_dir = data_dir
        # train
        self.train_data = None
        lst = os.path.join(data_dir, 'train', 'image_list.txt')
        if os.path.exists(lst):
            with open(lst, 'r') as f:
                self.train_data = [(os.path.join('train', items[0]), int(items[1]))
                                   for line in f.read().splitlines() for items in [line.split()]]
        # validation
        self.val_data = None
        lst = os.path.join(data_dir, 'validation', 'image_list.txt')
        if os.path.exists(lst):
            with open(lst, 'r') as f:
                self.val_data = [(os.path.join('validation', items[0]), int(items[1]))
                                  for line in f.read().splitlines() for items in [line.split()]]
            print('Warning: No test data found')
        # test
        self.test_data = None
        lst = os.path.join(data_dir, 'test', 'image_list.txt')
        if os.path.exists(lst):
            with open(lst, 'r') as f:
                self.test_data = [(os.path.join('test', line),)  for line in f.read().splitlines()]
        else:
            print('Warning: No test data found')

    def convert(self,
                tfrecords_path, 
                save_image_in_records=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, split in zip(['train', 'val', 'test'], [self.train_data, self.val_data, self.test_data]):    
            if split is None: 
                continue
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
                    if name == 'train':
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
                 data_dir='',
                 resize=None,
                 verbose=False):
        """Init a Loader object.
        
        
        Args:
            `save_image_in_records` (bool): If True, the image was saved in the record, otherwise only the image path was.
            `data_dir` (str): If save_image_in_records is False, append this string to the image_path saved in the record.
            `resize` (int): If given, resize the image to the given size
        """
        self.save_image_in_records = save_image_in_records
        self.data_dir = data_dir
        self.image_resize = resize
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
            image = tf.decode_raw(parsed_features['image'], tf.uint8)
            shape = tf.stack([parsed_features['width'], parsed_features['height'], 3], axis=0)
            image = tf.reshape(image, shape)
        else:
            filename = tf.decode_base64(parsed_features['image'])
            parsed_features['image_path'] = tf.identity(filename, name='image_path')
            image = tf.read_file(self.data_dir + filename)
            image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize image
        if self.image_resize is not None:
            image = tf.image.resize_images(image, (self.image_resize, self.image_resize))  
        parsed_features['image'] = tf.identity(image, name='image')
        # Class
        parsed_features['class'] = tf.to_int32(parsed_features['class'], name='class')
        # Return
        del parsed_features['height']
        del parsed_features['width']
        if self.verbose:
            print('\u001b[36mOutputs:\u001b[0m')
            print('\n'.join('   \u001b[46m%s\u001b[0m: %s' % (key, parsed_features[key]) 
                            for key in sorted(parsed_features.keys())))
        return parsed_features