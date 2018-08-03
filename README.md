![license](https://img.shields.io/github/license/ameroyer/TFDatasets.svg)
![GitHub repo size in bytes](https://img.shields.io/github/repo-size/ameroyer/TFDatasets.svg)
![GitHub top language](https://img.shields.io/github/languages/top/ameroyer/TFDatasets.svg)
![Maintenance](https://img.shields.io/maintenance/yes/2018.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/ameroyer/TFDatasets.svg)

# TFDatasets

`TFDatasets` is a collection of scripts to preprocess various Computer Vision datasets and convert them to `TFRecords` for easy integration in the `tf.data.Dataset` pipeline. The code was designed with Python 3+ and tensorflow 1.4+ in mind.

The notebook `load_datasets.ipynb` displays examples of writing and parsing TFRecords for each dataset. See the last section of this readme for an index of available datasets.

The notebook `preprocess.ipynb` displays of example of various preprocessing utilities for `tf.data.Dataset` (adding random crops, occlusion generation, subsampling etc.) demonstrated on the mnist dataset.

---


### Table of Contents

| Dataset | Link | Example | TFRecords contents |
| :-----: | :--: | :-----: | :----------------: |
| ACwS | [Apparel Classification with Style](http://www.vision.ee.ethz.ch/~lbossard/projects/accv12/index.html) | ![acws_thumb](images/acws.png) | image, class |
| CelebA | [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | ![celeba_thumb](images/celeba.png) | image, bounding-box, attributes, landmarks |
| CartoonSet | [CartoonSet](https://google.github.io/cartoonset/) | ![cartoonset_thumb](images/cartoonset.png) | image, bounding-box, attributes |
| CIFAR-10(0) | [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) | ![cifar10_thumb](images/cifar10.png) | image, class, (coarse_class), (coarse_)class-name, |
| Fashion MNIST| [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) | ![fashion_mnist_thumb](images/fashion_mnist.png) | image, class, index|
| MNIST | [MNIST](http://yann.lecun.com/exdb/mnist/) | ![mnist_thumb](images/mnist.png) | image, digit-class, index |
| MNIST-M | [MNIST-M](http://yaroslav.ganin.net/) | ![mnistm_thumb](images/mnistm.png) | image, digit-class, index |
| M2NIST | [M2NIST](https://www.kaggle.com/farhanhubble/multimnistm2nist/home) | ![m2nist_thumb](images/m2nist.png) | image, segmentation-mask, index |
| PACS | [PACS](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017) | ![pacs_thumb](images/pacs.png) | image, content-class, style-class |
| SVHN | [SVHN](http://ufldl.stanford.edu/housenumbers/) | ![svhn_thumb](images/svhn.png) | image, digit-class, index |
| Tiny ImageNet | [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) | ![tiny_imagenet_thumb](images/tiny_imagenet.png) | image, class, class-name, bounding-box |
| VisDA | [VisDA](http://ai.bu.edu/visda-2017/) | ![visda_thumb](images/visda.png) | image, class |


---

### Notes on the API

The specific module for any given dataset `data` is contained in `dataset_utils.data.py`. It contains a converter, `DataConverter` and a loader `DataLoader`.



#### Converter
The converter contains one method, `convert` that generates the TFRecords in the given `target_path`. Additionally, this method takes as keyword argument `sort` (defaults to `False`); If this is `True`, the entries in the TFRecords will be sorted by class labels when possible (e.g. classification task). Note that this means the `shuffle_buffer` size should be at least equal to the number of samples in the dataset for proper shuffling (hence it is not optimal for large datasets), but this can be a convenient feature to quickly filter/sample the dataset based on classes.


#### Loader
The loader simply builds a proper parsing function to extract data from the TFRecords and format it correctly. Such a function can then be passed to the `tf.data.Dataset` API map function.
