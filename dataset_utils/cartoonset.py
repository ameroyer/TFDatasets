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
from .tfrecords_utils import *
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
    bbox = np.array([xmin / w, ymin / h, xmax / w, ymax / h], dtype=np.float32)
    return img, bbox


"""Define features to be stored in the TFRecords"""
CartoonSetFeatures = Features([
    ('image', FeatureType.BYTES, FeatureLength.FIXED, (), None),
    ('bounding_box', FeatureType.FLOAT, FeatureLength.FIXED, (4,), None),
    ('chin_length', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('eye_angle', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('eye_color', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('eye_eyebrow_distance', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('eye_lashes', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('eye_lid', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('eye_slant', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('eyebrow_shape', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('eyebrow_thickness', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('eyebrow_weight', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('eyebrow_width', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('face_color', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('face_shape', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('facial_hair', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('glasses', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('glasses_color', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('hair', FeatureType.INT, FeatureLength.FIXED, (), None),
    ('hair_color', FeatureType.INT, FeatureLength.FIXED, (), None)
                                ])


class CartoonSetConverter(Converter):
    features = CartoonSetFeatures

    def __init__(self, data_dir):
        """Initialize the object for the CartoonSet dataset in `data_dir`"""
        self.data_dir = data_dir
        self.folders = [x for x in sorted(os.listdir(data_dir)) if x.isdigit()]
        if len(self.folders) == 0:
            self.folders = ['']
        print('Found %d data folders:' % len(self.folders))
        print('\n'.join('  %s' % os.path.join(data_dir, x) for x in self.folders))

    def convert(self,
                tfrecords_path,
                train_split=[0, 1, 2, 3, 4, 5, 6],
                val_split=[7],
                test_split=[8, 9],
                compression_type=None,
                save_image_in_records=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        # assert sets do not overlap
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
            writer = self.init_writer(writer_path, compression_type=compression_type)
            # For each directory
            print('\nLoad', name)
            for i, s in enumerate(split):
                if s >= len(self.folders)  or s < 0:
                    continue

                # for each image
                folder = self.folders[s]
                image_files = [x for x in os.listdir(os.path.join(self.data_dir, folder)) if x.endswith('.png')]
                for j, image_path in enumerate(image_files):
                    print('\r  folder %d/%d: image %d/%d' % (i + 1, len(split), j + 1, len(image_files)), end='')
                    # Image
                    full_image_path = os.path.join(self.data_dir, folder, image_path)
                    if save_image_in_records:
                        img, bbox = get_image_alpha_crop(full_image_path)
                        img = (img[:, :, :3] * 255.).astype(np.uint8).tostring()
                    else:
                        _, bbox = get_image_alpha_crop(full_image_path)
                        img = base64.b64encode(os.path.join(folder, image_path).encode('utf-8'))

                    # Attributes
                    csv_file = os.path.join(self.data_dir, folder, '%s.csv' % image_path.rsplit('.', 1)[0])
                    with open(csv_file, 'r') as f:
                        reader = csv.reader(f)
                        attributes = {row[0]: [int(row[1])] for row in reader}
                        attributes = [attributes[k] for k in sorted(attributes.keys())]
                        
                    # Write
                    writer.write(self.create_example_proto([img],
                                                           bbox.flatten(),
                                                           *attributes))

            # End
            writer.close()
            print('\nWrote %s in file %s (%.2fMB)' % (
                name, writer_path, os.path.getsize(writer_path) / 1e6))


class CartoonSetLoader(Loader):
    features = CartoonSetFeatures
    attributes_values_list = [('chin_length', 3), ('eye_angle', 3), ('eye_color', 5), ('eye_eyebrow_distance', 3),
                              ('eye_lashes', 2), ('eye_lid', 2), ('eye_slant', 3), ('eyebrow_shape', 14),
                              ('eyebrow_thickness', 4), ('eyebrow_weight', 2),  ('eyebrow_width', 3),
                              ('face_color', 11), ('face_shape', 7), ('facial_hair', 15), ('glasses', 12),
                              ('glasses_color', 7), ('hair', 111), ('hair_color', 10)]

    def __init__(self,
                 save_image_in_records=False,
                 image_dir='',
                 crop_images=False,
                 keep_crop_aspect_ratio=True,
                 image_size=None,
                 one_hot_attributes=False,
                 verbose=False):
        """Init a Loader object.

        Args:
            `save_image_in_records` (bool): If True, the image was saved in the record, otherwise only the image path was.
            `image_dir` (str): If save_image_in_records is False, append this string to the image_path saved in the record.
            `crop_images` (bool): If True, the image is cropped to the bounding box (remove transparent pixels)
            `keep_crop_aspect_ratio` (bool): If True, the cropping operation additionally pads the image to preserve
                the original ratio if necessary
            `image_size` (int): If given, resize the image to the given size
            `one_hot_attributes`: If True, the attributes are one-hot encoded, otherwise there are discrete values
        """
        self.save_image_in_records = save_image_in_records
        self.image_dir = image_dir
        self.crop_images = crop_images
        self.keep_crop_aspect_ratio = keep_crop_aspect_ratio
        self.image_size = image_size
        self.one_hot_attributes = one_hot_attributes
        self.verbose = verbose

    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Parse
        parsed_features = self.raw_parsing_fn(example_proto)
        # Reshape
        if self.save_image_in_records:
            image = decode_raw_image(parsed_features['image'], (500, 500, 3), image_size=None)
        else:
            filename = tf.decode_base64(parsed_features['image'])
            parsed_features['image_path'] = tf.identity(filename, name='image_path')
            image = decode_relative_image(filename, self.image_dir, image_size=None)
        # Crop image to bounding box
        if self.crop_images:
            bounding_box = parsed_features['bounding_box']
            if self.keep_crop_aspect_ratio:
                bounding_box = make_square_bounding_box(bounding_box, mode='max')
            image = tf.image.crop_and_resize(
                tf.expand_dims(image, axis=0), tf.expand_dims(bounding_box, axis=0), [0], (500, 500))[0]
            del parsed_features['bounding_box']
        # Resize image after cropping
        if self.image_size is not None:
            image = tf.image.resize_images(image, (self.image_size, self.image_size))
        parsed_features['image'] = tf.identity(image, name='image')
        # One-hot encode each attribute
        if self.one_hot_attributes:
            for key, num_values in self.attributes_values_list:
                parsed_features[key] = tf.one_hot(parsed_features[key], num_values, axis=-1, name='one_hot_%s' % key)
        # Return
        if self.verbose: print_records(parsed_features)
        return parsed_features
