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
    if type(model) is list:
        features = np.hstack(f(data))
    else:
        features = f(data)
    return features


def load_model(output_folder):
    with open(join(output_folder,'final_model.pkl'), 'rb') as save_file:
        l_encode = pickle.load(save_file)
    return l_encode
    

def visualize_features(W,W_prev,img_shape=(20,49)):
    # http://deeplearning.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity
    
    W_adjusted = np.dot(W_prev,W)
    neurons = np.reshape(W_adjusted,(img_shape[0],img_shape[1],W_adjusted.shape[1]))
    
    image = Image.fromarray(
                utils.tile_raster_images(X=W_adjusted.T,
                                   img_shape=(img_shape[0],img_shape[1]), tile_shape=(10, 10),
                                   tile_spacing=(1, 1)))
#    plt.imshow(image)
    return image

def train_autoencoder_layer(data,output_folder,layer_num,batch_size,learning_rate,num_epochs,
                                      momentum_flag,momentum,layer_size,corruption_level,nonlinearity,W_prev):

    if layer_num > 0 and len(W_prev) == 0:
        print('Please provide previous layer W')
        return

    model_folder = join(output_folder,'models')
    plot_folder = join(output_folder,'plots')

    # Create model
    if nonlinearity == 'rectify':
        nonlinearity_function = lasagne.nonlinearities.rectify
    elif nonlinearity == 'sigmoid':
        nonlinearity_function = lasagne.nonlinearities.sigmoid
    else:
        print('Nonlinearity not implemented: ',nonlinearity)
        return
    
    # Generate the model
    if layer_num == 0:
        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.tensor4('targets')
        l_in = lasagne.layers.InputLayer((None, data.shape[1], data.shape[2],data.shape[3]),input_var=input_var)
        l_flatten = lasagne.layers.ReshapeLayer(l_in,([0],-1))
        l_dropout = lasagne.layers.DropoutLayer(l_flatten,p=corruption_level,rescale=False)
        l_encode = lasagne.layers.DenseLayer(l_dropout, num_units=layer_size,nonlinearity=nonlinearity_function)
        l_decode = lasagne.layers.InverseLayer(l_encode,l_encode)
        l_final = lasagne.layers.ReshapeLayer(l_decode,([0],data.shape[1], data.shape[2],data.shape[3]))
#        l_final = lasagne.layers.InverseLayer(l_decode,l_flatten)
    else:
        input_var = T.matrix('inputs')
        target_var = T.matrix('targets')
        l_in = lasagne.layers.InputLayer((None, data.shape[1]),input_var=input_var)
        l_dropout = lasagne.layers.DropoutLayer(l_in,p=corruption_level,rescale=False)
        l_encode = lasagne.layers.DenseLayer(l_dropout, num_units=layer_size,nonlinearity=nonlinearity_function)
        l_decode = lasagne.layers.InverseLayer(l_encode,l_encode)
        l_final = lasagne.layers.ReshapeLayer(l_decode,([0],data.shape[1]))

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
    
    # Loop through epochs and train the model
    train_errs = np.zeros((num_epochs))
    val_errs = np.zeros((num_epochs))
    
    epoch = 0
    best_val_err_epoch = 0
    while epoch < best_val_err_epoch + 20 and epoch < num_epochs:
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
                
        if epoch % 50 == 0:
        
            # Save this model as a snapshot
            save_file = open(join(model_folder,'model_L{:02d}_E{:05d}.pkl'.format(layer_num,epoch)), 'wb')
            pickle.dump(l_encode,save_file,-1)
            pickle.dump(epoch, save_file ,-1)
            save_file.close()
            
            image = visualize_features(l_encode.W.get_value(),W_prev)
            image.save(join(plot_folder,'neuron_weights_L{:02d}_E{:05d}.png'.format(layer_num,epoch)))
        
        if epoch % 10 == 0:
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
                plt.savefig(join(output_folder,'reconstructionError_L{:02d}.png'.format(layer_num)))
                plt.close()

                # Save the best model
                save_file = open(join(output_folder,'best_model_L{:02d}.pkl'.format(layer_num)), 'wb')
                pickle.dump(l_encode,save_file,-1)
                pickle.dump(epoch, save_file ,-1)
                save_file.close()
        epoch+=1
        
    # Plot and save final error
    plt.plot(np.arange(epoch),train_errs[0:epoch],label='Training Error')
    plt.plot(np.arange(epoch),val_errs[0:epoch],label='Validation Error')
    plt.legend(loc='upper right')
    plt.savefig(join(output_folder,'reconstructionError_L{:02d}.png'.format(layer_num)))
    plt.close()
    
    save_file = open(join(output_folder,'errors_L{:02d}.pkl'.format(layer_num)), 'wb')
    pickle.dump(train_errs[0:epoch],save_file,-1)
    pickle.dump(val_errs[0:epoch],save_file,-1)
    save_file.close()
          
    best_model_file = join(output_folder,'best_model_L{:02d}.pkl'.format(layer_num))            
    with open(best_model_file, 'rb') as save_file:
        l_encode = pickle.load(save_file)
    

    return l_encode.W.get_value(), l_encode.b.get_value()
    
def train_autoencoder(data,output_folder,batch_size=20,learning_rate=0.01,num_epochs=10000,
                                      momentum_flag=True,momentum=0.9,layer_sizes = [50,50,50],
                                      corruption_levels=[0.3,0.3,0.3],nonlinearity='sigmoid'):    
    
    num_layers = len(layer_sizes)
    
    # Generate description file
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    with open(join(output_folder,'model_description.txt'), 'w') as f:
        f.write('\\n\\n=====Model Description=====\\n')
        f.write('model_description:' + 'dense_dense_dense' + '\\n')
        f.write('num_layers:' + str(num_layers) + '\\n')
        f.write('layer_sizes:' + str(layer_sizes) + '\\n')
        f.write('corruption_levels:' + str(corruption_levels) + '\\n')
        f.write('nonlinearity:' + nonlinearity + '\\n')
        f.write('learning_rate:' + str(learning_rate) + '\\n')
        f.write('momentum_flag:' + str(momentum_flag) + '\\n')
        if momentum_flag:
            f.write('momentum:' + str(momentum) + '\\n')
        else:
            f.write('momentum:' + 'NA' + '\\n')


    # Prepare model folder
    model_folder = join(output_folder,'models')
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    plot_folder = join(output_folder,'plots')
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
        
    # Define nonlinearity
    if nonlinearity == 'rectify':
        nonlinearity_function = lasagne.nonlinearities.rectify
    elif nonlinearity == 'sigmoid':
        nonlinearity_function = lasagne.nonlinearities.sigmoid
    else:
        print('Nonlinearity not implemented: ',nonlinearity)
        return
    
    # Generate input layers of final model
    l_in = lasagne.layers.InputLayer((None, data.shape[1], data.shape[2],data.shape[3]))
    l_prev = lasagne.layers.ReshapeLayer(l_in,([0],-1))
    
    # Train each dense layer of the model separately
    layer_data = data
    W_prev = np.identity(lasagne.layers.get_output_shape(l_prev)[1]) # Keep track of multiplied weights for plotting
    for m in np.arange(num_layers):
        W_layer, b_layer = train_autoencoder_layer(layer_data,output_folder,m,batch_size=batch_size,
                                                   learning_rate=learning_rate,num_epochs = num_epochs, 
                                                   momentum_flag = momentum_flag, momentum = momentum,
                                                   layer_size = layer_sizes[m], 
                                                   corruption_level = corruption_levels[m],
                                                   nonlinearity = nonlinearity,W_prev = W_prev)
        l_dense = lasagne.layers.DenseLayer(l_prev,W_layer.shape[1],W=W_layer,b=b_layer,
                                            nonlinearity=nonlinearity_function)
        W_prev = np.dot(W_prev,W_layer)
        layer_data = generate_features_from_model(data,l_dense)
        if m == 0:
            l_encode = [l_dense]
        else:
            l_encode.append(l_dense)
        print(l_encode)
        l_prev = l_dense
    
        
    # Save the best model
    save_file = open(join(output_folder,'final_model.pkl'), 'wb')
    pickle.dump(l_encode,save_file,-1)
    save_file.close()