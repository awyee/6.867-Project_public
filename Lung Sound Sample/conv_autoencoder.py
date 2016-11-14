# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 07:14:49 2016

@author: daniel
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:53:45 2016

@author: Daniel
"""

import numpy as np
import pickle
import theano
import theano.tensor as T
import lasagne
from os.path import join
from sklearn import cross_validation
import matplotlib.pyplot as plt
import os
import time
import PIL.Image as Image
import utils
import scipy


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
     

def generate_features_from_model(data,model):
    input_var = T.tensor4('inputs')
    f = theano.function([input_var], lasagne.layers.get_output(model,inputs=input_var,deterministic=True))
    features = f(data)
    return features

def load_model(output_folder):
    with open(join(output_folder,'final_model.pkl'), 'rb') as save_file:
        l_encode = pickle.load(save_file)
    return l_encode
    

def visualize_reconstruction(visualization_data,l_final,buffer_pixels=5):
    input_var = T.tensor4('inputs')
    f = theano.function([input_var], lasagne.layers.get_output(l_final,inputs=input_var,deterministic=True))
    reconstructed_data = np.squeeze(f(visualization_data))
    
    visualization_data = np.squeeze(visualization_data)
    num_images = visualization_data.shape[0]
    img_height = visualization_data.shape[1]
    img_width = visualization_data.shape[2]
    output_height = img_height*2+buffer_pixels
    output_width = num_images*img_width + (num_images-1)*buffer_pixels
    output_image = np.zeros((output_height,output_width))

    # Create an image tiling the original data and reconstructions
    for img_num in np.arange(num_images):
        output_image[0:img_height,img_num*(img_width+buffer_pixels):(img_num+1)*(img_width+buffer_pixels)-buffer_pixels] = visualization_data[img_num,:,:]
        output_image[img_height+buffer_pixels:,img_num*(img_width+buffer_pixels):(img_num+1)*(img_width+buffer_pixels)-buffer_pixels] = reconstructed_data[img_num,:,:]
    return output_image


def train_conv_autoencoder(data,output_folder,img_shape,batch_size=20,
                           learning_rate=0.01,num_epochs=10000,
                           momentum_flag=True,momentum=0.9,
                           nonlinearity='sigmoid',
                           conv_num_filters=100,conv_filter_size=3,
                           pool_pool_size=2,encode_size=40):

    # Generate description file
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    with open(join(output_folder,'model_description.txt'), 'w') as f:
        f.write('\\n\\n=====Model Description=====\\n')
        f.write('model_description:' + 'conv_conv_pool_conv_conv_pool_dense' + '\\n')
        f.write('nonlinearity:' + nonlinearity + '\\n')
        f.write('learning_rate:' + str(learning_rate) + '\\n')
        f.write('momentum_flag:' + str(momentum_flag) + '\\n')
        if momentum_flag:
            f.write('momentum:' + str(momentum) + '\\n')
        else:
            f.write('momentum:' + 'NA' + '\\n')
        f.write('conv_num_filters:' + str(conv_num_filters) + '\\n')
        f.write('conv_filter_size:' + str(conv_filter_size) + '\\n')
        f.write('pool_pool_size:' + str(pool_pool_size) + '\\n')
        f.write('encode_size:' + str(encode_size) + '\\n')


    # Prepare model folder
    model_folder = join(output_folder,'models')
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    plot_folder = join(output_folder,'plots')
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)

    # Create model
    if nonlinearity == 'rectify':
        nonlinearity_function = lasagne.nonlinearities.rectify
    elif nonlinearity == 'sigmoid':
        nonlinearity_function = lasagne.nonlinearities.sigmoid
    else:
        print('Nonlinearity not implemented: ',nonlinearity)
        return    
    
    
    # Define theano symbolic variables    
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    
    # Create network
    l_in = lasagne.layers.InputLayer((None, data.shape[1], data.shape[2],data.shape[3]),input_var=input_var)
    l_conv1 = lasagne.layers.Conv2DLayer(l_in,conv_num_filters,conv_filter_size)
    l_conv2 = lasagne.layers.Conv2DLayer(l_conv1,conv_num_filters,conv_filter_size)
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv2,pool_pool_size)
    l_conv3 = lasagne.layers.Conv2DLayer(l_pool1,conv_num_filters,conv_filter_size)
    l_conv4 = lasagne.layers.Conv2DLayer(l_conv3,conv_num_filters,conv_filter_size)
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv4,pool_pool_size)
    pool_shape = lasagne.layers.get_output_shape(l_pool2)
    l_flatten = lasagne.layers.ReshapeLayer(l_pool2,([0],-1))
    l_encode = lasagne.layers.DenseLayer(l_flatten,encode_size,nonlinearity=nonlinearity_function)
    l_decode = lasagne.layers.InverseLayer(l_encode,l_encode)
#    l_unflatten = lasagne.layers.ReshapeLayer(l_decode,([0],conv_num_filters,(img_shape[0]-conv_filter_size+1)//2,(img_shape[1]-conv_filter_size+1)//2))
    l_unflatten = lasagne.layers.ReshapeLayer(l_decode,([0],pool_shape[1],pool_shape[2],pool_shape[3]))
    l_unpool2 = lasagne.layers.InverseLayer(l_unflatten,l_pool2)
    l_deconv4 = lasagne.layers.InverseLayer(l_unpool2,l_conv4)
    l_deconv3 = lasagne.layers.InverseLayer(l_deconv4,l_conv3)    
    l_unpool1 = lasagne.layers.InverseLayer(l_deconv3,l_pool1)
    l_deconv2 = lasagne.layers.InverseLayer(l_unpool1,l_conv2)
    l_deconv1 = lasagne.layers.InverseLayer(l_deconv2,l_conv1)
    l_final = l_deconv1
    
     # Generate training loss expression
    prediction = lasagne.layers.get_output(l_final)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = lasagne.objectives.aggregate(loss,mode='mean')
    
    # Generate validation loss expression
    val_prediction = lasagne.layers.get_output(l_final,deterministic='True')
    val_loss = lasagne.objectives.squared_error(val_prediction, target_var)
    val_loss = lasagne.objectives.aggregate(val_loss,mode='mean')
    
    # Generate update expression
    params = lasagne.layers.get_all_params(l_final, trainable=True)
    if momentum_flag:
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=momentum)
    else:
        updates = lasagne.updates.sgd(loss, params, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], val_loss)
    
    # Split data into training and validation sets
    x_train, x_test = cross_validation.train_test_split(data,test_size=0.25)
    
    # Create reconstruction data
    visualization_data = np.concatenate((x_train[0:5,:,:,:],x_test[0:5,:,:,:]))
    
    # Loop through epochs and train the model
    train_errs = np.zeros((num_epochs))
    val_errs = np.zeros((num_epochs))
    
    epoch = 0
    best_val_err_epoch = 0
    while epoch < best_val_err_epoch + 10 and epoch < num_epochs:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(x_train, x_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(x_test, x_test, batch_size, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1
        
        train_errs[epoch] = train_err/train_batches
        val_errs[epoch] = val_err/val_batches

        # Define best validation error
        if epoch == 0:
            best_val_err = val_err
            best_val_err_epoch = epoch
                
        if epoch % 5 == 0:
        
            # Save this model as a snapshot
            save_file = open(join(model_folder,'model_E{:05d}.pkl'.format(epoch)), 'wb')
            pickle.dump(l_encode,save_file,-1)
            pickle.dump(epoch, save_file ,-1)
            save_file.close()
            
            image = visualize_reconstruction(visualization_data,l_final)
            plt.imsave(fname=join(plot_folder,'reconstructions_E{:05d}.png'.format(epoch)),arr=image)
        
        if epoch % 1 == 0:
            print('Epoch {}, training cost {}'.format(epoch, train_errs[epoch]))
            print('Epoch {}, validation cost {}'.format(epoch, val_errs[epoch]))
            print('Seconds elapsed: {:0.0f}'.format(time.time()-start_time))
            
            if val_err <= best_val_err:
                best_val_err = val_err
                best_val_err_epoch = epoch

                # Plot error so far
                plt.plot(np.arange(epoch+1),train_errs[0:epoch+1],label='Training Error')
                plt.plot(np.arange(epoch+1),val_errs[0:epoch+1],label='Validation Error')
                plt.legend(loc='upper right')
                plt.savefig(join(output_folder,'reconstructionError.png'))
                plt.close()

                # Save the best model
                save_file = open(join(output_folder,'final_model.pkl'), 'wb')
                pickle.dump(l_encode,save_file,-1)
                pickle.dump(epoch, save_file ,-1)
                save_file.close()
        epoch+=1
    
    # Plot and save final error
    plt.plot(np.arange(epoch),train_errs[0:epoch],label='Training Error')
    plt.plot(np.arange(epoch),val_errs[0:epoch],label='Validation Error')
    plt.legend(loc='upper right')
    plt.savefig(join(output_folder,'reconstructionError.png'))
    plt.close()
    
    save_file = open(join(output_folder,'errors.pkl'), 'wb')
    pickle.dump(train_errs[0:epoch],save_file,-1)
    pickle.dump(val_errs[0:epoch],save_file,-1)
    save_file.close()
          
    best_model_file = join(output_folder,'final_model.pkl')            
    with open(best_model_file, 'rb') as save_file:
        l_encode = pickle.load(save_file)
    
