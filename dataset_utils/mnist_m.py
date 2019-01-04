from __future__ import print_function
##############################
#      MNIST-M dataset       #
# http://yaroslav.ganin.net/ #
##############################
import os
import numpy as np

from matplotlib import image as mpimg
import tensorflow as tf

from .tfrecords_utils import *
from .mnist import MNISTLoader, MNISTFeatures


class MNISTMConverter(Converter):
    features = MNISTFeatures

    def __init__(self, data_dir):
        """Initialize the object for the MNIST-M dataset in `data_dir`"""
        print('Loading original MNIST-M data from', data_dir)
        self.data = []
        for name in ['train', 'test']:
            split = os.path.join(data_dir, 'mnist_m_%s_labels.txt' % name)
            image_dir = os.path.join(data_dir, 'mnist_m_%s' % name)
            if not os.path.isfile(split):
                print('Warning: Missing %s data' % name)
            elif not os.path.exists(image_dir):
                print('Warning: Missing %s image directory' % name)
            else:
                with open(split, 'r') as f:
                    images, labels = zip(*[line.split() for line in f.read().splitlines()])
                    labels = list(map(int, labels))
                    self.data.append((name, image_dir, images, labels))

    def convert(self, tfrecords_path, compression_type=None, sort=False):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`"""
        for name, image_dir, images, labels in self.data:
            # Init writer
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = self.init_writer(writer_path, compression_type=compression_type)
            num_items = len(labels)
            labels_order = np.argsort(labels, axis=0) if sort else range(num_items)
            for x, index in enumerate(labels_order):
                print('\rLoad %s: %d / %d' % (name, x + 1, num_items), end='')
                img = np.ceil(255. * mpimg.imread(os.path.join(image_dir, images[index])))
                img = img.astype(np.uint8)
                class_id = labels[index]
                writer.write(self.create_example_proto([class_id], [img.tostring()], [index]))
            # End writing
            writer.close()
            print('\nWrote %s in file %s (%.2fMB)' % (
                name, writer_path, os.path.getsize(writer_path) / 1e6))
            print()


MNISTMLoader = MNISTLoader
MNISTMLoader.shape = (32, 32, 3)
