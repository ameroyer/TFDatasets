# TFDatasets

A set of scripts to convert various Computer Vision datasets to TFRecords.
Also show small examples of how to easily integrate them in the tf.data.Dataset pipeline (TF 1.4+)


### General API
**Note:** this code was tested with Python 3+ and tensorflow 1.4+

`load_datasets.ipynb` is the main notebook displaying examples of writing and loading TFRecords for the datasets. The specific module for a given dataset `data` is contained in `dataset_utils.data.py`. It contains a converter, `DataConverter` and a loader `DataLoader`.


#### Converter
The converter contains one method, `convert` that generates the TFRecords in the given `target_path`. Additionally, this method takes as keyword argument `sort` (defaults to `False`); If this is `True`, the entries in the TFRecords will be sorted by class labels when possible (e.g. classification task). Note that this means the `shuffle_buffer` size should be at least equal to the number of samples in the dataset for proper shuffling (hence it is not optimal for large datasets), but this can be a convenient feature to quickly filter/sample the dataset based on classes.


#### Loader
The loader simply builds a proper parsing function to extract data from the TFRecords and format it correctly. Such a function can then be passed to the `tf.data.Dataset` API map function.


### Table of Contents

| Dataset | Link | Example | Preprocessed Data |
| ------- | ---- | ------ | --- |
| MNIST | [MNIST](http://yann.lecun.com/exdb/mnist/) | ![mnist_thumb](images/mnist.png) | image, digit-class, index |
| SVHN | [SVHN](http://ufldl.stanford.edu/housenumbers/) | ![svhn_thumb](images/svhn.png) | image, digit-class, index |
| MNIST-M | [MNISTM](http://yaroslav.ganin.net/) | ![mnistm_thumb](images/mnistm.png) | image, digit-class, index |