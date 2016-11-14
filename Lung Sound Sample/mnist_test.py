# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:05:49 2016

@author: Daniel
"""

import os
import theano
import theano.tensor as T
import pickle
import gzip
import numpy as np
import autoencoder
import conv_autoencoder

def load_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    dataset = 'data/mnist/mnist.pkl.gz'
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

#    def shared_dataset(data_xy, borrow=True):
#        """ Function that loads the dataset into shared variables
#
#        The reason we store our dataset in shared variables is to allow
#        Theano to copy it into the GPU memory (when code is run on GPU).
#        Since copying data into the GPU is slow, copying a minibatch everytime
#        is needed (the default behaviour if the data is not in a shared
#        variable) would lead to a large decrease in performance.
#        """
#        data_x, data_y = data_xy
#        shared_x = theano.shared(np.asarray(data_x,
#                                               dtype=theano.config.floatX),
#                                 borrow=borrow)
#        shared_y = theano.shared(np.asarray(data_y,
#                                               dtype=theano.config.floatX),
#                                 borrow=borrow)
#        # When storing data on the GPU it has to be stored as floats
#        # therefore we will store the labels as ``floatX`` as well
#        # (``shared_y`` does exactly that). But during our computations
#        # we need them as ints (we use labels as index, and if they are
#        # floats it doesn't make sense) therefore instead of returning
#        # ``shared_y`` we will have to cast it to int. This little hack
#        # lets ous get around this issue
#        return shared_x, T.cast(shared_y, 'int32')
#
#    test_set_x, test_set_y = shared_dataset(test_set)
#    valid_set_x, valid_set_y = shared_dataset(valid_set)
#    train_set_x, train_set_y = shared_dataset(train_set)
#
#    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
#            (test_set_x, test_set_y)]

    train_set_x = train_set[0]
    return train_set_x
    
def train_mnist_sda_model():
    train_set_x = load_data()
    train_set_x = np.reshape(train_set_x,(train_set_x.shape[0],28,28))
    train_set_x = np.expand_dims(train_set_x,1)
    output_folder = 'lung-sound-deep-learning-models/mnist/sda0'
    autoencoder.train_autoencoder(train_set_x,output_folder)
    
def train_mnist_conv_ae_model():
    train_set_x = load_data()
    train_set_x = np.reshape(train_set_x,(train_set_x.shape[0],28,28))
    train_set_x = np.expand_dims(train_set_x,1)
    output_folder = 'lung-sound-deep-learning-models/mnist/convae0'
    conv_autoencoder.train_conv_autoencoder(train_set_x,output_folder,(28,28))
    
if __name__ == '__main__':
#    train_mnist_sda_model()
    train_mnist_conv_ae_model()