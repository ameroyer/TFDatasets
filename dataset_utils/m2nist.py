from __future__ import print_function
#############################################################
#                       M2NIST dataset                      #
# https://www.kaggle.com/farhanhubble/multimnistm2nist/home #
#############################################################
import os
import numpy as np
import tensorflow as tf

from .tfrecords_utils import * 


class M2NISTConverter(Converter):
    
    def __init__(self, data_dir):
        """Initialize the object for the M2NIST dataset in `data_dir`"""
        print('Loading original Multidigit MNIST data from', data_dir)
        self.images = os.path.join(data_dir, 'combined.npy')
        self.masks = os.path.join(data_dir, 'segmented.npy')
        assert os.path.isfile(self.images)
        assert os.path.isfile(self.masks)

    def convert(self, tfrecords_path, train_split=0.7, val_split=0.1, test_split=0.2):
        """Convert the dataset in TFRecords saved in the given `tfrecords_path`
        If `sort` is True, the Example will be sorted by class in the final TFRecords.
        """
        # Create train-val-test split
        images = np.load(self.images)
        masks = np.load(self.masks, mmap_mode='r') # this is a large array !
        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)
        assert train_split + val_split + test_split == 1.0
        train_fence = int(train_split * len(indices))
        val_fence = int((train_split + val_split) * len(indices))
        test_fence = len(indices)
        # Write
        for name, start, end in [('train', 0, train_fence),
                                 ('val', train_fence, val_fence),
                                 ('test', val_fence, test_fence)]:  
            if start == end:          
                print('Warning: Empty %s split' % name)
                continue
            writer_path = '%s_%s' % (tfrecords_path, name)
            writer = tf.python_io.TFRecordWriter(writer_path)
            for i, index in enumerate(indices[start:end]):
                print('\rLoad %s: %d / %d' % (name, i + 1, end - start), end='')
                img = images[i].astype(np.uint8)
                mask = masks[i, :, :, :10].astype(np.float32)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'mask': bytes_feature([mask.tostring()]),
                    'image': bytes_feature([img.tostring()]),
                    'id': int64_feature([index])}))                
                writer.write(example.SerializeToString())
            writer.close()
            print('\nWrote %s in file %s' % (name, writer_path))
            print()
        
        
def viz_mask(mask):
    """Given a (batch, w, h, 10) array, returns a visualization"""
    rgb_palette = np.array([(248, 183, 205), (246, 210, 224),  (200, 231, 245),  (103, 163, 217),  (6, 113, 183),
                            (249, 200, 14), (248, 102, 36),  (234, 53, 70), (102, 46, 155), (67, 188, 205)])
    mask = np.expand_dims(mask, axis=-1)
    mask = np.tile(mask, (1, 1, 1, 1, 3))
    mask = np.sum(mask * rgb_palette, axis=-2)
    mask = np.clip(mask, 0, 255)
    return mask     
            
class M2NISTLoader():
    
    def __init__(self, image_size=None, verbose=False):
        """Init a Loader object. Loaded images will be resized to size `image_size`."""
        self.image_size = image_size
        self.verbose = verbose                
    
    def parsing_fn(self, example_proto):
        """tf.data.Dataset parsing function."""
        # Basic features
        features = {'mask': tf.FixedLenFeature((), tf.string),
                    'image': tf.FixedLenFeature((), tf.string),
                    'id': tf.FixedLenFeature((), tf.int64)}      
        parsed_features = tf.parse_single_example(example_proto, features)  
        # Image
        image = decode_raw_image(parsed_features['image'], (64, 84, 1), image_size=self.image_size)
        image = tf.identity(image, name='image')
        # Mask
        mask  = tf.decode_raw(parsed_features['mask'], tf.float32)
        mask = tf.reshape(mask, (64, 84, 10))
        if self.image_size is not None:
            mask = tf.image.resize_images(mask, (self.image_size, self.image_size), method=tf.image.ResizeMethod.BILINEAR)
            mask = tf.to_float(mask > 0.5)
        # Index
        index = tf.to_int32(parsed_features['id'], name='index')
        # Return
        records_dict = {'image': image, 'mask': mask, 'id': index}
        if self.verbose: print_records(records_dict)
        return records_dict