from __future__ import print_function
######################################################
#                    PACS dataset                    #
# http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017 #
######################################################
import base64
import csv
import os
import numpy as np
from matplotlib import image as mpimg
from .tfrecords_utils import *
import tensorflow as tf


class PACSConverter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the PACS dataset in `data_dir`"""
        self.data_dir = data_dir
        self.raw_data = [[[] for _ in range(7)] for _ in range(4)]
        # For each style (4)
        for i, style in enumerate(['art_painting', 'cartoon', 'photo', 'sketch']):
            style_dir = os.path.join(self.data_dir, style)
            if not os.path.exists(style_dir):
                print('Warning: no directory found for style %s' % style)
                continue
            # For each class (7)
            for j, content in enumerate(['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']):
                content_dir = os.path.join(self.data_dir, style, content)
                if not os.path.exists(content_dir):
                    print('Warning: no directory found for content %s in style %s' % (content, style))
                    continue                
                self.raw_data[i][j] = sorted([os.path.join(style, content, x)
                                              for x in os.listdir(content_dir) if x.rsplit('.', 1)[1] in ['jpg', 'png']])
        self.train_data = None
        self.val_data = None
        self.test_data = None  
        self.has_generated_split = False
                
                
    def generate_split(self, split_path, train=0.7, val=0.1, test=0.2):
        """Generate a train, val, test split uniformly over all classes 
           and export the result as a text file in split_path (0 = train, 1 = val, 2 = test"""
        assert train + val + test == 1.0
        self.train_data = [[[] for _ in range(7)] for _ in range(4)]
        self.val_data = [[[] for _ in range(7)] for _ in range(4)]
        self.test_data = [[[] for _ in range(7)] for _ in range(4)]
        # Uniform split
        for style, d in enumerate(self.raw_data):            
            for content, image_paths in enumerate(d):
                n = len(image_paths)
                train_fence = int(n * train)
                val_fence = train_fence + int(n * val)
                paths = np.array(image_paths)
                indices = np.array(list(range(n)), dtype=np.int)
                np.random.shuffle(indices)
                self.train_data[style][content] = paths[indices[:train_fence]]
                self.val_data[style][content] = paths[indices[train_fence:val_fence]]
                self.test_data[style][content] = paths[indices[val_fence:]]
        # Export to file
        with open(split_path, 'w') as f:
            f.write('\n'.join('%s 0' % image for style_list in self.train_data for content_list in style_list 
                              for image in content_list))
            f.write('\n'.join('%s 1' % image for style_list in self.val_data for content_list in style_list 
                              for image in content_list))
            f.write('\n'.join('%s 2' % image for style_list in self.test_data for content_list in style_list 
                              for image in content_list))
        self.has_generated_split = True
        print('Splits saved in', split_path)
            

    def convert(self, tfrecords_path, save_image_in_records=False, separate_styles=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        # If no split has been generated, then convert the full data
        if self.has_generated_split:
            data = zip(['train', 'val', 'test'], [self.train_data, self.val_data, self.test_data])
        else:
            data = zip(['full'], [self.raw_data])
        style_names = ['art_painting', 'cartoon', 'photo', 'sketch']
            
        for name, split in data:    
            if split is None: 
                continue
            if not separate_styles:
                writer_path = '%s_%s' % (tfrecords_path, name)
                writer = tf.python_io.TFRecordWriter(writer_path)
            # For each dir
            print('\nLoad', name)
            for s, style_list in enumerate(split):
                if separate_styles:
                    writer_path = '%s_%s_%s' % (tfrecords_path, style_names[s], name)
                    writer = tf.python_io.TFRecordWriter(writer_path)
                for c, content_list in enumerate(style_list):
                    print('\rstyle %d/%d - content %d/%d' % (s + 1, len(split), c + 1, len(style_list)), end='')
                    for image_path in content_list:
                        feature = {}
                        # Image
                        if save_image_in_records:
                            img = mpimg.imread(os.path.join(self.data_dir, image_path))
                            if style_names[s] == 'sketch':
                                img = img * 255.
                                img = img[:, :, :3]
                            feature['image'] = bytes_feature([img.astype(np.uint8).tostring()])
                        else:
                            feature['image'] = bytes_feature([base64.b64encode(image_path.encode('utf-8'))])
                        # Class
                        feature['class_style'] = int64_feature([s])
                        feature['class_content'] = int64_feature([c])
                        # Write
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())
                if separate_styles:
                    writer.close()
                    print('\nWrote %s for style %s in file %s' % (name, style_names[s], writer_path))
            if  not separate_styles:
                writer.close()
                print('\nWrote %s in file %s' % (name, writer_path))
        print()
            
            
class PACSLoader():
    style_names = ['art_painting', 'cartoon', 'photo', 'sketch']
    content_names = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    
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
                    'class_content': tf.FixedLenFeature((), tf.int64),
                    'class_style': tf.FixedLenFeature((), tf.int64)
                   }     
        parsed_features = tf.parse_single_example(example_proto, features)   
        # Load image
        if self.save_image_in_records: 
            image = decode_raw_image(parsed_features['image'], (227, 227, 3), image_size=self.image_size)
        else:
            filename = tf.decode_base64(parsed_features['image'])
            parsed_features['image_path'] = tf.identity(filename, name='image_path')
            image = decode_relative_image(filename, self.image_dir, image_size=self.image_size)
        parsed_features['image'] = tf.identity(image, name='image')
        # Class
        parsed_features['class_content'] = tf.to_int32(parsed_features['class_content'])
        parsed_features['class_style'] = tf.to_int32(parsed_features['class_style'])
        # Return
        if self.verbose: print_records(parsed_features)
        return parsed_features