from abc import ABC, abstractmethod
import tensorflow as tf

### Convenienve function for writing Feature in TFRecords
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


### Base converter class
class Converter(ABC):
    @abstractmethod
    def __init__(self, data_dir):
        pass
    
    @abstractmethod
    def convert(self, tfrecords_path, sort=True):
        pass
    
### Base loader class
class Loader(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def parsing_fn(self, example_proto):
        pass