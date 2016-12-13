import tensorflow as tf
import numpy as np
from DataSet import DataSet
from six.moves import cPickle as pickle
import sys
import math
import time

Writetofile=False
Augment=False
invariance = False
layer2_switch = False
layer3_switch = False

# Hyperparameters
batch_size = 100
learning_rate = 0.01
num_training_steps = 4301

layer1_filter_size = 4
layer1_depth = 16
layer1_stride = 1
layer2_filter_size = 844
layer2_depth = 32
layer2_stride = 2
layer3_filter_size = 5
layer3_stride = 2
layer3_depth = 64
layer3_num_hidden = 8
startloc=400
# layer4_num_hidden = 64

# Add max pooling
pooling = True
layer1_pool_filter_size = 2
layer1_pool_stride = 2
layer2_pool_filter_size = 2
layer2_pool_stride = 2

# Enable dropout and weight decay normalization
dropout_prob = 1  # set to < 1.0 to apply dropout, 1.0 to remove
weight_penalty = 0.01  # set to > 0.0 to apply weight penalty, 0.0 to remove

OUTPUT_FILE = 'ConvResults.csv'
NUM_CHANNELS = 1
NUM_LABELS = 1
NUM_FEATS=20


class PCGNet:
    def __init__(self):
        '''Initialize the class by loading the required datasets
		and building the graph'''
        self.load_pickled_dataset(DATA_FILE)
        self.graph=tf.Graph()
        self.define_tensorflow_graph()

    def define_tensorflow_graph(self):
        print ('\nDefining model...')

        with self.graph.as_default():

            tf_train_batch = tf.placeholder(shape=[batch_size, NUM_FEATS], dtype=tf.float32)
            tf_valid_dataset = tf.constant(self.val_X)
            tf_test_dataset = tf.constant(self.test_X)
            tf_train_dataset = tf.placeholder(
                tf.float64, shape=[len(self.train_X), NUM_FEATS])
            tf_train_labels = tf.placeholder(shape=[batch_size, NUM_LABELS], dtype=tf.float32)
            prediction_grid = tf.placeholder(shape=[None, NUM_FEATS], dtype=tf.float32)

            b = tf.Variable(tf.random_normal(shape=[1, batch_size]))

            gamma = tf.constant(-50.0)

            # Implement dropout
            dropout_keep_prob = tf.placeholder(tf.float32)

            # Network weights/parameters that will be learned
            layer1_weights = tf.Variable(tf.truncated_normal(
                [NUM_FEATS, layer1_depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([layer1_depth]))
            layer1_feat_map_size = int(math.ceil(float(NUM_FEATS) / layer1_stride))
            if pooling:
                layer1_feat_map_size = int(math.ceil(float(NUM_FEATS) / layer1_pool_stride))

            if layer2_switch:
                layer2_weights = tf.Variable(tf.truncated_normal(
                    [layer1_depth, layer2_depth], stddev=0.1))
                layer2_biases = tf.Variable(tf.constant(1.0, shape=[layer2_depth]))
                layer2_feat_map_size = int(math.ceil(float(layer1_feat_map_size) / layer2_stride))
                if pooling:
                    layer2_feat_map_size = int(math.ceil(float(layer2_feat_map_size) / layer2_pool_stride))

                if layer3_switch:
                    layer3a_weights = tf.Variable(tf.truncated_normal(
                        [layer2_depth, layer3_depth], stddev=0.1))
                    layer3a_biases = tf.Variable(tf.constant(1.0, shape=[layer3_depth]))
                    layer3a_feat_map_size = int(math.ceil(float(layer2_feat_map_size) / layer3_stride))

                    layer3_weights = tf.Variable(tf.truncated_normal(
                        [layer3_depth, layer3_num_hidden], stddev=0.1))
                    layer3_biases = tf.Variable(tf.constant(1.0, shape=[layer3_num_hidden]))

                else:
                    layer3_weights = tf.Variable(tf.truncated_normal(
                        [layer2_depth, layer3_num_hidden], stddev=0.1))
                    layer3_biases = tf.Variable(tf.constant(1.0, shape=[layer3_num_hidden]))
            else:
                layer3_weights = tf.Variable(tf.truncated_normal(
                    [layer1_depth, layer3_num_hidden], stddev=0.1))
                layer3_biases = tf.Variable(tf.constant(1.0, shape=[layer3_num_hidden]))

            layer4_weights = tf.Variable(tf.truncated_normal(
                [layer3_num_hidden, NUM_LABELS], stddev=0.1))
            layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

            # Model
            def network_model(data):

                dist = tf.reduce_sum(tf.square(tf_train_batch), 1)
                dist = tf.reshape(dist, [-1, 1])
                sq_dists = tf.add(tf.sub(dist, tf.mul(2., tf.matmul(tf_train_batch, tf.transpose(tf_train_batch)))),
                                  tf.transpose(dist))
                my_kernel = tf.exp(tf.mul(gamma, tf.abs(sq_dists)))
                '''Define the actual network architecture'''

                # Layer 1
                #conv1 = tf.nn.convolution(data, layer1_weights, [1, layer1_stride, layer1_stride, 1], padding='SAME')
                #hidden = tf.nn.relu(conv1 + layer1_biases)
                data=tf.cast(data, tf.float32)
                hidden = tf.nn.relu(tf.matmul(data, layer1_weights) + layer1_biases)

                # if pooling:
                #     hidden = tf.nn.max_pool(hidden, ksize=[1, layer1_pool_filter_size, layer1_pool_filter_size, 1],
                #                             strides=[1, layer1_pool_stride, layer1_pool_stride, 1],
                #                             padding='SAME', name='pool1')

                # Layer 2
                if layer2_switch:
                    hidden = tf.nn.relu(tf.matmul(hidden, layer2_weights) + layer2_biases)

                    # if pooling:
                    #     hidden = tf.nn.max_pool(hidden, ksize=[1, layer2_pool_filter_size, layer2_pool_filter_size, 1],
                    #                             strides=[1, layer2_pool_stride, layer2_pool_stride, 1],
                    #                             padding='SAME', name='pool2')

                    if layer3_switch:
                        #conv3 = tf.nn.convolution(hidden, layer3a_weights, [1, layer3_stride, layer3_stride, 1], padding='SAME')
                        #hidden = tf.nn.relu(conv3 + layer3a_biases)
                        hidden = tf.nn.relu(tf.matmul(hidden, layer3_weights) + layer3_biases)

                # Layer 3
                #shape = hidden.get_shape().as_list()
                #reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(hidden, layer3_weights) + layer3_biases)
                hidden = tf.nn.dropout(hidden, dropout_keep_prob)

                # Layer 4
                output = tf.matmul(hidden, layer4_weights) + layer4_biases
                return output

            # Training computation
            logits = network_model(tf_train_batch)
            shape = tf_train_labels.get_shape().as_list()
            new_tf_labels=tf.reshape(tf_train_labels, [shape[0], 1])
            loss = tf.reduce_sum(abs(tf.nn.sigmoid(logits)-tf_train_labels))

            # Add weight decay penalty
            loss = loss + weight_decay_penalty([layer1_weights, layer3_weights, layer4_weights], weight_penalty)

            # Optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            # Predictions for the training, validation, and test data.
            batch_prediction = tf.nn.sigmoid(logits)
            valid_prediction = tf.nn.sigmoid(network_model(tf_valid_dataset))
            test_prediction = tf.nn.sigmoid(network_model(tf_test_dataset))
            train_prediction = tf.nn.sigmoid(network_model(tf_train_dataset))

            def train_model(num_steps=num_training_steps):
                '''Train the model with minibatches in a tensorflow session'''
                with tf.Session(graph=self.graph) as session:
                    tf.initialize_all_variables().run()
                    print ('Initializing variables...')

                    for step in range(num_steps):
                        offset = (step * batch_size+startloc) % (self.train_Y.shape[0] - batch_size)
                        batch_data = self.train_X[offset:(offset + batch_size), :]
                        batch_labels = self.train_Y[offset:(offset + batch_size)]

                        # Data to feed into the placeholder variables in the tensorflow graph
                        feed_dict = {tf_train_batch: batch_data, tf_train_labels: batch_labels,
                                     dropout_keep_prob: dropout_prob}
                        _, l, predictions = session.run(
                            [optimizer, loss, batch_prediction], feed_dict=feed_dict)
                        if (step % 200 == 0):
                            train_preds = session.run(train_prediction, feed_dict={tf_train_dataset: self.train_X,
                                                                                   dropout_keep_prob: 1.0})
                            val_preds = session.run(valid_prediction, feed_dict={dropout_keep_prob: 1.0})
                            test_preds = session.run(test_prediction, feed_dict={dropout_keep_prob: 1.0})
                            print('')
                            print('Batch loss at step %d: %f' % (step, l))
                            print('Batch training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                            print('Validation accuracy: %.1f%%' % accuracy(val_preds, self.val_Y))
                            print('Full train accuracy: %.1f%%' % accuracy(train_preds, self.train_Y))
                            #if accuracy(val_preds, self.val_Y)>70:
                            #    break
                            print('Test accuracy: %.1f%%' % accuracy(test_preds, self.test_Y))

                    # This code is for the final question

                    # if self.invariance:
                    #     print ("\n Obtaining final results on invariance sets!")
                    #     sets = [self.val_X, self.translated_val_X, self.bright_val_X, self.dark_val_X,
                    #             self.high_contrast_val_X, self.low_contrast_val_X, self.flipped_val_X,
                    #             self.inverted_val_X, ]
                    #     set_names = ['normal validation', 'translated', 'brightened', 'darkened',
                    #                  'high contrast', 'low contrast', 'flipped', 'inverted']
                    #
                    #     for i in range(len(sets)):
                    #         preds = session.run(test_prediction,
                    #                             feed_dict={tf_test_dataset: sets[i], dropout_keep_prob: 1.0})
                    #         print ('Accuracy on', set_names[i], 'data: %.1f%%' % accuracy(preds, self.val_Y))
                    #
                    #         # save final preds to make confusion matrix
                    #         if i == 0:
                    #             self.final_val_preds = preds

                            # save train model function so it can be called later

                    print('')
                    print('Parameters')
                    if pooling:
                        print ('Pooling is ON')
                    else:
                        print ('Pooling is OFF')
                    print('Dropout Probability: %1f' % dropout_prob)
                    print('Weight Penalty: %1f' % weight_penalty)
                    print('Number of Steps: %d' % num_training_steps)

                    if Writetofile:
                        fd = open(OUTPUT_FILE, 'a')
                        fd.write('\n %d, %d, %d,' % (layer1_filter_size, layer1_depth, layer1_stride))
                        if layer2_switch:
                            fd.write(' %d, %d, %d,' % (layer2_filter_size, layer2_depth, layer2_stride))
                        else:
                            fd.write(' NaN, NaN, NaN,')
                        if layer3_switch:
                            fd.write(' %d, %d, %d,' % (layer3_filter_size, layer3_depth, layer3_stride))
                        else:
                            fd.write(' NaN, NaN, NaN,')
                        fd.write(' %d, NaN, ' % (layer3_num_hidden))
                        if pooling:
                            fd.write(' %d, %d, %d, %d,' % (layer1_pool_filter_size, layer1_pool_stride,layer2_pool_filter_size, layer2_pool_stride))
                        else:
                            fd.write(' NaN, NaN, NaN, NaN,')
                        fd.write('%r, %r, %r, %1f, %1f, %d, %.1f%%, %.1f%%' % (pooling, Augment, invariance, dropout_prob, weight_penalty, num_training_steps, accuracy(val_preds, self.val_Y), accuracy(train_preds, self.train_Y)))
                        fd.close()

            self.train_model = train_model

    def load_pickled_dataset(self, pickle_file):
        # with open(pickle_file, 'rb') as f:
        #     save = pickle.load(f)
        #     self.train_X = save['train_data']       #Learn PANDAS
        #     self.train_Y = save['train_labels']
        #     self.val_X = save['val_data']
        #     self.val_Y = save['val_labels']
        #
        #     if INCLUDE_TEST_SET:
        #         self.test_X = save['test_data']
        #         self.test_Y = save['test_labels']
        #     del save  # hint to help gc free up memory
        # Variables
        data = DataSet.load_data_set('Split Data_Standard_12-09-2016_auto_Normal_Abnormal')
        data_type = 'auto'
        y_label = 'Normal/Abnormal'  # 'Normal/Abnormal'

        ''' Logistic Regression with Auto Data'''
        # Data Preparation

        self.test_Y = data[0]/2+0.5
        self.train_Y = data[2]/2+0.5
        self.val_Y = data[4]/2+0.5

        Ydim=np.array(self.test_Y).shape
        self.test_Y=np.array(self.test_Y).reshape(Ydim[0],1)

        Ydim=np.array(self.train_Y).shape
        self.train_Y=np.array(self.train_Y).reshape(Ydim[0],1)

        Ydim=np.array(self.val_Y).shape
        self.val_Y=np.array(self.val_Y).reshape(Ydim[0],1)

        self.test_X = data[1]
        self.train_X = data[3]
        self.val_X = data[5]

        # Balance DataSets
        #test_y, test_x = DataSet.balance_dataset_by_reproduction(test_y, test_x)
        #train_y, train_x = DataSet.balance_dataset_by_reproduction(train_y, train_x)
        #val_y, val_x = DataSet.balance_dataset_by_reproduction(val_y, val_x)
        print ('Training set', self.train_X.shape, self.train_Y.shape)
        print ('Validation set', self.val_X.shape, self.val_Y.shape)
        print ('Test set', self.test_X.shape, self.test_Y.shape)


def weight_decay_penalty(weights, penalty):
    return penalty * sum([tf.nn.l2_loss(w) for w in weights])


def accuracy(predictions, labels):
    classifications= (predictions>0.5)
    return (100.0 * sum(classifications == labels)/ predictions.shape[0])
#    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
#            / predictions.shape[0])


if __name__ == '__main__':
#    tstart = time.time()

#    for i in range(10,20,1):
#        dropout_prob=i/float(20)
#        print("%d %1f" %(i,dropout_prob))
#    for i in range(100,2100,100):
#        num_training_steps=i
#    for i in range (-16,4,1):
#        weight_penalty=10**(i/float(2))
#    pooling=True
#    for i in range(2,11,2):
#    for i in (True,False):
#        pooling=i
#        layer1_pool_filter_size=i
#        layer1_pool_stride=i
#        for j in range(2,12,3):
#                    layer2_pool_filter_size=j
#                    layer2_pool_stride=j
#            layer2_filter_size=j
#            for k in range(1,5):
#                layer2_stride=k
#                for p in range(8,33,8):
#                    layer2_depth=p
#                    Augment=True
#                    num_training_steps=6001
#                    print('%r %d %d %d' % (i,j,k,p))
#                    print('%d %d' % (i,j))
                    t1 = time.time()
                    conv_net = PCGNet()
                    conv_net.train_model()
                    t2 = time.time()
                    print('Data File: %s' %DATA_FILE)
                    print ("Finished training. Total time taken:", t2 - t1)
#                    print "Total time elapsed:", t2 - tstart