from abc import ABC, abstractmethod
import tensorflow as tf

### Convenience function for writing Feature in TFRecords
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


### Convenience function for creating a tf.data.Dataset
def get_tf_dataset(path_to_tfrecords, parsing_fn, shuffle_buffer=1, batch_size=8):
    """Create a basic tensorflow Dataset object from a TFRecords.
    
    Args:
        path_to_tfrecords: Path to the TFrecords
        parsing_fn: parsing function to apply to every element (load Examples)
        shuffle_buffer: Shuffle buffer size to randomize the dataset
        batch_size: Batch size
    """
    print('Creating dataset with batch_size %d and shuffle buffer %d' % (
            batch_size, shuffle_buffer))
    data = tf.data.TFRecordDataset(path_to_tfrecords)
    data = data.shuffle(shuffle_buffer)
    data = data.map(parsing_fn)
    data = data.repeat()
    data = data.batch(batch_size)
    iterator = data.make_one_shot_iterator()
    in_ = iterator.get_next()
    return in_


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