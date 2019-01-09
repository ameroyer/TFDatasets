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
from .acws import ACWSLoader


VisdaFeatures = Features([
    ('image', FeatureType.BYTES, FeatureLength.FIXED, (), None),
    ('width', FeatureType.INT, FeatureLength.FIXED, (), tf.constant(-1, dtype=tf.int64)),
    ('height', FeatureType.INT, FeatureLength.FIXED, (), tf.constant(-1, dtype=tf.int64)),
    ('class', FeatureType.INT, FeatureLength.FIXED, (), None),
                        ])


class VisdaClassificationConverter(Converter):
    features = VisdaFeatures

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
                compression_type=None,
                save_image_in_records=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, split in self.data:
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = self.init_writer(writer_path, compression_type=compression_type)
            print('\nLoad', name)
            for i, aux in enumerate(split):
                print('\rImage %d/%d' % (i + 1, len(split)), end='')
                # Image
                image_path = aux[0]
                height, width = None, None
                if save_image_in_records:
                    img = mpimg.imread(os.path.join(self.data_dir, image_path))
                    if name == 'train': # synthetic
                        img = img * 255.
                    height = [img.shape[0]]
                    width = [img.shape[1]]
                    img = img.astype(np.uint8).tostring()
                else:
                    mpimg.imread('/home/aroyer/Documents/icon.png')
                    img = base64.b64encode(image_path.encode('utf-8'))
                # Class
                class_id = [aux[1]] if len(aux) > 1 else None
                # Write
                writer.write(self.create_example_proto([img], height, width, class_id))
            writer.close()
            print('\nWrote %s in file %s (%.2fMB)' % (
                name, writer_path, os.path.getsize(writer_path) / 1e6))
        print()


class VisdaClassificationLoader(ACWSLoader):
    classes_names = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
                     'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
