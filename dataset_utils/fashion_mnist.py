from __future__ import print_function
####################################################
#              Fashion MNIST dataset               #
# https://github.com/zalandoresearch/fashion-mnist #
####################################################
import codecs
import os
import numpy as np
import tensorflow as tf

from .tfrecords_utils import * 
from .mnist import MNISTConverter, MNISTLoader


class FashionMNISTConverter(MNISTConverter): 

    def __init__(self, data_dir):
        """Initialize the object for the fashion MNIST dataset in `data_dir`"""
        print('Loading original FashionMNIST data from', data_dir)
        self.data = []
        for name, key in [('train', 'train'), ('test', 't10k')]:
            images = os.path.join(data_dir, '%s-images-idx3-ubyte' % key)
            labels = os.path.join(data_dir, '%s-labels-idx1-ubyte' % key)
            if not os.path.isfile(images) or not os.path.isfile(labels):             
                print('Warning: Missing %s data' % name)
            else:
                self.data.append((name, images, labels))
            
            
class FashionMNISTLoader(MNISTLoader):
    classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
                     'Sneaker', 'Bag', 'Ankle boot']
    
    def __init__(self, image_size=None, verbose=False):
        super(FashionMNISTLoader, self).__init__(image_size=image_size, verbose=verbose)