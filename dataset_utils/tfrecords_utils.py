from abc import ABC, abstractmethod
from enum import Enum
import tensorflow as tf


### Convenience function for writing Feature in TFRecords
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


### Base Features Class
class FeatureType(Enum):
    INT = (tf.int64, int64_feature)
    FLOAT = (tf.float32, float_feature)
    BYTES = (tf.string, bytes_feature)
    
    
class FeatureLength(Enum):
    FIXED = 0
    VAR = 1    
    
    
class Features:
    def __init__(self, feature_list):
        """A feature_list is a list of tuple containing each feature name, type, whether it is of 
            fixed or variable length, its shape, and optional default value.
            Where type is one of 'int64', 'string' or 'float'."""
        # Features dictionnary (reading)
        self.features_read = {name: (tf.FixedLenFeature(shape, feature_type.value[0], default_value=default) 
                                     if feature_length == FeatureLength.FIXED else 
                                     tf.VarLenFeature(feature_type.value[0]))
                              for name, feature_type, feature_length, shape, default in feature_list}
        # Featured dictionnary (writing)
        self.features_write = [(name, feature_type.value[1]) for name, feature_type, _, _, _ in feature_list]
        

### Base converter class
class Converter(ABC):
    @property
    def features(self):
        raise NotImplementedError
        
    @abstractmethod
    def __init__(self, data_dir):
        pass
            
    def init_writer(self, writer_path, compression_type=None):
        """Returns the TFRecordWriter object writing to the given path"""
        assert compression_type in [None, 'gzip', 'zlib']
        writer_options = None
        if compression_type == 'gzip':
            writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        elif compression_type == 'zlib':
            writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        return tf.python_io.TFRecordWriter(writer_path, options=writer_options)
    
    def create_example_proto(self, *args):
        """Create a TFRecords example protobuffer from the given arguments. Ignore `None` values"""
        feature = {name: fn(x) for x, (name, fn) in zip(args, self.features.features_write) if x is not None}
        example = tf.train.Example(features=tf.train.Features(feature=feature))                
        return example.SerializeToString()
    
    @abstractmethod
    def convert(self, tfrecords_path, compression_type=None):
        """Convert the dataset to TFRecords format"""
        pass
    
    
### Base loader class
class Loader(ABC):
    @property
    def features(self):
        raise NotImplementedError
        
    @abstractmethod
    def __init__(self):
        pass
    
    def raw_parsing_fn(self, example_proto):
        return tf.parse_single_example(example_proto, self.features.features_read)  
    
    @abstractmethod
    def parsing_fn(self, example_proto):
        pass

    
### Other convenience functions for parsing
def get_tf_dataset(path_to_tfrecords,
                   parsing_fn, 
                   compression_type=None,
                   compression_buffer=0,
                   shuffle_buffer=1, 
                   batch_size=8):
    """Create a basic one-shot tensorflow Dataset object from a TFRecords.
    
    Args:
        path_to_tfrecords: Path to the TFrecords
        parsing_fn: parsing function to apply to every element (load Examples)
        shuffle_buffer: Shuffle buffer size to randomize the dataset
        batch_size: Batch size
    """
    assert compression_type in [None, 'gzip', 'zlib']
    if compression_type is None:
        data = tf.data.TFRecordDataset(path_to_tfrecords)
    else:
        data = tf.data.TFRecordDataset(path_to_tfrecords, 
                                       compression_type=compression_type.upper(), 
                                       buffer_size=compression_buffer)
    data = data.shuffle(shuffle_buffer)
    data = data.map(parsing_fn)
    data = data.batch(batch_size)
    iterator = data.make_one_shot_iterator()
    in_ = iterator.get_next()
    return in_


def get_sample(target_path, loader, compression_type=None, shuffle_buffer=1, batch_size=8):
    """Return data sample"""
    with tf.Graph().as_default():
        data = get_tf_dataset(
            '%s_train' % target_path, loader.parsing_fn, compression_type=compression_type,
            shuffle_buffer=shuffle_buffer, batch_size=batch_size)
        if 'bounding_box' in data:
            data['image'] = tf.image.draw_bounding_boxes(
                data['image'], tf.expand_dims(data['bounding_box'], axis=1))
        with tf.Session() as sess:
            return sess.run(data) 


def decode_raw_image(feature, shape, image_size=None):
    """Decode raw image
    Args:
        feature: raw image as a tf.String tensor
        shape: Shape of the raw image
        image_size: If given, resize the decoded image to the given square size
    Returns:
        The resized image
    """
    image = tf.decode_raw(feature, tf.uint8)
    image = tf.reshape(image, shape)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if image_size is not None:
        image = tf.image.resize_images(image, (image_size, image_size))
    return image


def decode_relative_image(filename, image_dir, image_size=None):
    """Decode image from a filename
    Args:
        feature: image path as a tf.String tensor (decoded)
        image_dir: Base image dir
        image_size: If given, resize the decoded image to the given square size
    Returns:
        The resized image
    """
    image = tf.read_file(image_dir + filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if image_size is not None:
        image = tf.image.resize_images(image, (image_size, image_size))
    return image
    
    
def make_square_bounding_box(bounding_box, mode='max'):
    """Given a bounding box [ymin, xmin, ymax, xmax] in [0., 1.], 
        compute a square bounding box centered around it,
        whose side is equal to the maximum or minimum side
    """
    assert mode in ['max', 'min']
    width = bounding_box[2] - bounding_box[0]
    height = bounding_box[3] - bounding_box[1]
    size = tf.maximum(width, height) if mode == 'max' else  tf.minimum(width, height)
    offset_x = (size - width) / 2.
    offset_y = (size - height) / 2.
    offset = tf.stack([- offset_x, - offset_y, offset_x, offset_y], axis=0)
    return bounding_box + offset


def print_records(records_dict):
    """Pretty-print a dictionary for verbose mode"""
    print('\u001b[36mContents:\u001b[0m')
    print('\n'.join('   \u001b[46m%s\u001b[0m: %s' % (key, records_dict[key]) 
                    for key in sorted(records_dict.keys())))
