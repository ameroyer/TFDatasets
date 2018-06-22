from __future__ import print_function
########################################
#           CartoonSet dataset         #
# https://google.github.io/cartoonset/ #
########################################
import base64
import csv
import os
import numpy as np
from matplotlib import image as mpimg
from .tfrecords_utils import Converter, _bytes_feature, _floats_feature, _int64_feature 
import tensorflow as tf


def get_image_alpha_crop(image_path):
    """Given the path to an image with four channels (eg .png), 
        returns the bounding box englobing all its visible pixels."""
    img = mpimg.imread(image_path)
    assert len(img.shape) == 3
    assert img.shape[-1] == 4
    w, h, _ = img.shape
    transparent = np.where(img[:, :, -1] > 0.)
    xmin = np.amin(transparent[0])
    ymin = np.amin(transparent[1])
    xmax = np.amax(transparent[0])
    ymax = np.amax(transparent[1])
    bbox = np.array([xmin / w, ymin / h, xmax / w, ymax /h], dtype=np.float32)
    return img, bbox


class CartoonSetConverter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the CartoonSet dataset in `data_dir`"""
        self.data_dir = data_dir
        self.folders = [x for x in sorted(os.listdir(data_dir)) if x.isdigit()]
        print('Found %d data folders:' % len(self.folders))
        print('\n'.join('  %s' % os.path.join(data_dir, x) for x in self.folders))

    def convert(self,
                tfrecords_path, 
                train_split=[0, 1, 2, 3, 4, 5, 6],
                val_split=[7], 
                test_split=[8, 9],
                save_image_in_records=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        # assert set do not overlap
        set_train_split = set(train_split)
        set_val_split = set(val_split)
        set_test_split = set(test_split)
        assert not len(set_train_split.intersection(set_val_split))
        assert not len(set_train_split.intersection(set_test_split))
        assert not len(set_val_split.intersection(set_test_split))
        # parse
        for name, split in [['train', train_split], ['val', val_split], ['test', test_split]]:    
            if not(len(split)):          
                print('No split selected for %s data' % name)
                continue
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            # For each dir
            print('\nLoad', name)
            for i, s in enumerate(split):
                if s >= len(self.folders)  or s < 0:
                    continue
                # for each image
                folder = self.folders[s]
                image_files = [x for x in os.listdir(os.path.join(self.data_dir, folder)) if x.endswith('.png')]
                for j, image_path in enumerate(image_files):
                    print('\r  folder %d/%d: image %d/%d' % (i + 1, len(split), j + 1, len(image_files)), end='') 
                    feature = {}
                    # Image
                    full_image_path = os.path.join(self.data_dir, folder, image_path)
                    if save_image_in_records:
                        img, bbox = get_image_alpha_crop(full_image_path)
                        img =(img[:, :, :3] * 255.).astype(np.uint8)
                        feature['image'] = _bytes_feature([img.tostring()])
                    else:
                        _, bbox = get_image_alpha_crop(full_image_path)
                        feature['image'] = _bytes_feature([
                                    base64.b64encode(os.path.join(folder, image_path).encode('utf-8'))])
                    # Bounding Box
                    feature['bounding_box'] = _floats_feature(bbox.flatten())
                    # Attributes
                    csv_file = os.path.join(self.data_dir, folder, '%s.csv' % image_path.rsplit('.', 1)[0])
                    with open(csv_file, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            feature[row[0]] = _int64_feature([int(row[1])])
                    # Write
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
            print()
            
            
class CartoonSetLoader():
    
    def __init__(self,
                 save_image_in_records=False, 
                 data_dir='',
                 crop_images=False, 
                 keep_crop_aspect_ratio=True, 
                 resize=None,
                 one_hot_attributes=False):
        """Init a Loader object.
        
        
        Args:
            `save_image_in_records` (bool): If True, the image was saved in the record, otherwise only the image path was.
            `data_dir` (str): If save_image_in_records is False, append this string to the image_path saved in the record.
            `crop_images` (bool): If True, the image is cropped to the bounding box (remove transparent pixels)
            `keep_crop_aspect_ratio` (bool): If True, the cropping operation additionally pads the image to preserve 
                the original ratio if necessary
            `resize` (int): If given, resize the image to the given size
            `one_hot_attributes`: If True, the attributes are one-hot encoded
        """
        self.save_image_in_records = save_image_in_records
        self.data_dir = data_dir
        self.crop_images = crop_images
        self.keep_crop_aspect_ratio = keep_crop_aspect_ratio
        self.image_resize = resize
        self.one_hot_attributes = one_hot_attributes
    
    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Basic features
        features = {'image' : tf.FixedLenFeature((), tf.string),
                    'bounding_box': tf.FixedLenFeature((4,), tf.float32),
                    'chin_length': tf.FixedLenFeature((), tf.int64),
                    'eye_angle': tf.FixedLenFeature((), tf.int64),
                    'eye_color': tf.FixedLenFeature((), tf.int64),
                    'eye_eyebrow_distance': tf.FixedLenFeature((), tf.int64),
                    'eye_lashes': tf.FixedLenFeature((), tf.int64),
                    'eye_lid': tf.FixedLenFeature((), tf.int64),
                    'eye_slant': tf.FixedLenFeature((), tf.int64),
                    'eyebrow_shape': tf.FixedLenFeature((), tf.int64),
                    'eyebrow_thickness': tf.FixedLenFeature((), tf.int64),
                    'eyebrow_weight': tf.FixedLenFeature((), tf.int64),
                    'eyebrow_width': tf.FixedLenFeature((), tf.int64),
                    'face_color': tf.FixedLenFeature((), tf.int64),
                    'face_shape': tf.FixedLenFeature((), tf.int64),
                    'facial_hair': tf.FixedLenFeature((), tf.int64),
                    'glasses': tf.FixedLenFeature((), tf.int64),
                    'glasses_color': tf.FixedLenFeature((), tf.int64),
                    'hair': tf.FixedLenFeature((), tf.int64),
                    'hair_color': tf.FixedLenFeature((), tf.int64),
                   }     
        parsed_features = tf.parse_single_example(example_proto, features)   
        # Load image
        if self.save_image_in_records: 
            image = tf.decode_raw(parsed_features['image'], tf.uint8)
        else:
            filename = tf.decode_base64(parsed_features['image'])
            parsed_features['image_path'] = filename
            image = tf.read_file(self.data_dir + filename)
            image = tf.image.decode_png(image, channels=4)
            image = image[:, :, :3]
        image = tf.reshape(image, (500, 500, 3))
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Crop image
        if self.crop_images:
            bounding_box = parsed_features['bounding_box']
            if self.keep_crop_aspect_ratio:                
                width = bounding_box[2] - bounding_box[0]
                height = bounding_box[3] - bounding_box[1]
                size = tf.maximum(width, height)
                offset_x = (size - width) / 2.
                offset_y = (size - height) / 2.
                offset = tf.stack([- offset_x, - offset_y, offset_x, offset_y], axis=0)
                bounding_box += offset
            image = tf.image.crop_and_resize(tf.expand_dims(image, axis=0),
                                             tf.expand_dims(bounding_box, axis=0),
                                             [0], (500, 500))[0]
            del parsed_features['bounding_box']
        # Resize image
        if self.image_resize is not None:
            image = tf.image.resize_images(image, (self.image_resize, self.image_resize))  
        parsed_features['image'] = image
        # One-hot encode each attribute
        if self.one_hot_attributes:
            for key, num_values in [('chin_length', 3),
                                    ('eye_angle', 3),
                                    ('eye_color', 5),
                                    ('eye_eyebrow_distance', 3),
                                    ('eye_lashes', 2),
                                    ('eye_lid', 2),
                                    ('eye_slant', 3),
                                    ('eyebrow_shape', 14),
                                    ('eyebrow_thickness', 4),
                                    ('eyebrow_weight', 2),
                                    ('eyebrow_width', 3),
                                    ('face_color', 11),
                                    ('face_shape', 7),
                                    ('facial_hair', 15),
                                    ('glasses', 12),
                                    ('glasses_color', 7),
                                    ('hair', 111),
                                    ('hair_color', 10)]:
                parsed_features[key] = tf.one_hot(parsed_features[key], num_values, axis=-1)
        # Return
        return parsed_features