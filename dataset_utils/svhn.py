from __future__ import print_function
###########################################
#               SVHN dataset              #
# http://ufldl.stanford.edu/housenumbers/ #
###########################################
import os
import numpy as np
import scipy.io
import tensorflow as tf

from .tfrecords_utils import *
from .mnist import MNISTLoader, MNISTFeatures


class SVHNConverter(Converter):
    features = MNISTFeatures
    
    def __init__(self, data_dir):
        """Initialize the object for the SVHN dataset in `data_dir`"""
        print('Loading original SVHN data from', data_dir)
        self.data = []
        for name, key in [('train', 'train'), ('val', 'extra'), ('test', 'test')]:
            data = os.path.join(data_dir, '%s_32x32.mat' % key)
            if not os.path.isfile(data):             
                print('Warning: Missing %s data' % name)
            else:
                self.data.append((name, data))

    def convert(self, tfrecords_path, compression_type=None, sort=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, data in self.data:
            # Load
            mat = scipy.io.loadmat(data)
            images, labels = mat['X'], mat['y']
            num_items = labels.shape[0]
            # Write
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = self.init_writer(writer_path, compression_type=compression_type)
            labels_order = np.argsort(labels, axis=0) if sort else range(num_items)
            for x, index in enumerate(labels_order):
                print('\rLoad %s: %d / %d' % (name, x + 1, num_items), end='')
                img = images[:, :, :, index]
                img = img.astype(np.uint8)
                class_id = int(labels[index, 0])
                class_id = 0 if class_id == 10 else class_id
                writer.write(self.create_example_proto([class_id], [img.tostring()], [index]))
            # End
            writer.close()
            print('\nWrote %s in file %s (%.2fMB)' % (
                name, writer_path, os.path.getsize(writer_path) / 1e6))
            print()
            

SVHNLoader = MNISTLoader
SVHNLoader.shape = (32, 32, 3)