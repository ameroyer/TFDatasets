from __future__ import print_function
####################################################
#                 CelebA dataset                   #
# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html #
####################################################
import base64
import csv
import os
import numpy as np
from matplotlib import image as mpimg
from .tfrecords_utils import *
import tensorflow as tf


def load_bounding_boxes(file_path):
    annots = {}
    with open(file_path, 'r') as f:
        for line in f.read().splitlines()[2:]:
            path, xmin, ymin, width, height = line.split()
            annots[path] = np.array([float(xmin), 
                                     float(ymin), 
                                     float(xmin) + float(width),
                                     float(ymin) + float(height)])
    return annots


def load_attributes(file_path):
    annots = {}
    with open(file_path, 'r') as f:
        for line in f.read().splitlines()[2:]:
            values = line.split()
            annots[values[0]] = np.array(list(map(float, values[1:])))
    return annots


def load_landmarks(file_path):
    annots = {}
    with open(file_path, 'r') as f:
        for line in f.read().splitlines()[2:]:
            values = line.split()
            landmarks = np.array(list(map(float, values[1:])))
            landmarks = np.reshape(landmarks, (-1, 2))
            annots[values[0]] = landmarks
    return annots
    

class CelebaConverter(Converter):
    
    def __init__(self, data_dir, cropped_and_aligned=True):
        """Initialize the object for the CelebA dataset in `data_dir`"""
        # train/val/test partition
        self.data_dir = data_dir
        self.splits = [[], [], []]
        with open(os.path.join(self.data_dir, 'list_eval_partition.txt'), 'r') as f:
            for line in f.read().splitlines():
                path, split = line.split()
                self.splits[int(split)].append(path)
        # aligned and cropped version
        if cropped_and_aligned:
            self.image_dir = os.path.join('Img', 'img_align_celeba')
            self.bounding_boxes = None
            self.attributes = load_attributes(os.path.join(self.data_dir, 'Anno', 'list_attr_celeba.txt'))
            self.landmarks = load_landmarks(os.path.join(self.data_dir, 'Anno', 'list_landmarks_align_celeba.txt'))
        # unprocessed version
        else:
            self.image_dir = os.path.join('Img', 'img_celeba')
            self.bounding_boxes = load_bounding_boxes(os.path.join(self.data_dir, 'Anno', 'list_bbox_celeba.txt'))
            self.attributes = load_attributes(os.path.join(self.data_dir, 'Anno', 'list_attr_celeba.txt'))
            self.landmarks = load_landmarks(os.path.join(self.data_dir, 'Anno', 'list_landmarks_celeba.txt'))

    def convert(self,
                tfrecords_path, 
                save_image_in_records=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, split in zip(['train', 'val', 'test'], self.splits):    
            if not(len(split)):          
                print('Empty split for %s data' % name)
                continue
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            # For each dir
            print('\nLoad', name)
            for i, path in enumerate(split):
                print('\rImage %d/%d' % (i + 1, len(split)), end='')
                feature = {}
                # Image
                image_path = os.path.join(self.image_dir, path)
                img = mpimg.imread(os.path.join(self.data_dir, image_path))
                height, width = img.shape[:2]
                if save_image_in_records:
                    feature['image'] = bytes_feature([img.astype(np.uint8).tostring()])
                    feature['width'] = int64_feature([img.shape[0]])
                    feature['height'] = int64_feature([img.shape[1]])
                else:
                    feature['image'] = bytes_feature([base64.b64encode(image_path.encode('utf-8'))])
                # Bounding Box (in [0, 1])
                if self.bounding_boxes is not None:
                    bounding_box = self.bounding_boxes[path] / np.array([width, height, width, height])
                    feature['bounding_box'] = floats_feature(bounding_box)
                # Attributes (in {-1, 1})
                feature['attributes'] = floats_feature(self.attributes[path])
                # Landmarks (in [0, 1])
                landmarks = self.landmarks[path] / np.array([width, height])
                feature['landmarks'] = floats_feature(landmarks.flatten())
                # Write
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
        print()
            
            
class CelebaLoader():
    attributes_list = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 
        'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    
    def __init__(self,
                 save_image_in_records=False, 
                 data_dir='',
                 resize=None,
                 verbose=False):
        """Init a Loader object.
        
        
        Args:
            `save_image_in_records` (bool): If True, the image was saved in the record, otherwise only the image path was.
            `data_dir` (str): If save_image_in_records is False, append this string to the image_path saved in the record.
            `crop_images` (bool): If True, the image is cropped to the bounding box (remove transparent pixels)
            `keep_crop_aspect_ratio` (bool): If True, the cropping operation additionally pads the image to preserve 
                the original ratio if necessary
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
                    'bounding_box': tf.FixedLenFeature((4,), tf.float32, default_value=tf.constant([0., 0., 1., 1.])),
                    'attributes': tf.FixedLenFeature((40,), tf.float32),
                    'landmarks': tf.FixedLenFeature((5, 2), tf.float32),
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
        # Attributes to Boolean
        parsed_features['attributes'] = tf.greater(parsed_features['attributes'], 0., name='attributes')
        # Return
        del parsed_features['height']
        del parsed_features['width']
        if self.verbose:
            print('\u001b[36mOutputs:\u001b[0m')
            print('\n'.join('   \u001b[46m%s\u001b[0m: %s' % (key, parsed_features[key]) 
                            for key in sorted(parsed_features.keys())))
        return parsed_features