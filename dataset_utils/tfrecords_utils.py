from abc import ABC, abstractmethod
import tensorflow as tf

### Convenience function for writing Feature in TFRecords
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


### Convenience function for creating a basic tf.data.Dataset
def get_tf_dataset(path_to_tfrecords, parsing_fn, shuffle_buffer=1, batch_size=8):
    """Create a basic one-shot tensorflow Dataset object from a TFRecords.
    
    Args:
        path_to_tfrecords: Path to the TFrecords
        parsing_fn: parsing function to apply to every element (load Examples)
        shuffle_buffer: Shuffle buffer size to randomize the dataset
        batch_size: Batch size
    """
    print('[dataset] batch_size = %d, shuffle buffer = %d' % (
            batch_size, shuffle_buffer))
    data = tf.data.TFRecordDataset(path_to_tfrecords)
    data = data.shuffle(shuffle_buffer)
    data = data.map(parsing_fn)
    data = data.batch(batch_size)
    iterator = data.make_one_shot_iterator()
    in_ = iterator.get_next()
    return in_


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


def print_records(records_dict):
    """Print a dictionnary for verbose mode"""
    print('\u001b[36mOutputs:\u001b[0m')
    print('\n'.join('   \u001b[46m%s\u001b[0m: %s' % (key, records_dict[key]) 
                    for key in sorted(records_dict.keys())))
    
    
def make_square_bounding_box(bounding_box, mode='max'):
    """Given a bounding box [ymin, xmin, ymax, xmax] in [0., 1.], compute a square bounding box centered around it,
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
        

### Base converter class
class Converter(ABC):
    @abstractmethod
    def __init__(self, data_dir):
        pass
    
    @abstractmethod
    def convert(self, tfrecords_path):
        pass
    
### Base loader class
class Loader(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def parsing_fn(self, example_proto):
        pass