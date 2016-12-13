import tensorflow as tf
import numpy as np
from DataSet import DataSet
from six.moves import cPickle as pickle
from sklearn.decomposition import PCA
import sys
import math
import time

balanced=False

UsePCA=False
Writetofile=True
Augment=False
invariance = False
layer2_switch = False
layer3_switch = False

layer0_depth= np.array([20, 5, 5, 5, 5, 5])
layer0_size= np.array([20, 10, 10, 10, 10, 10])

# Hyperparameters
batch_size = 100
learning_rate = 0.01
num_training_steps = 3401

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
PCAfeats=20
# layer4_num_hidden = 64

# Add max pooling
pooling = True
layer1_pool_filter_size = 2
layer1_pool_stride = 2
layer2_pool_filter_size = 2
layer2_pool_stride = 2

# Enable dropout and weight decay normalization
dropout_prob = 0.9  # set to < 1.0 to apply dropout, 1.0 to remove
weight_penalty = 0.01  # set to > 0.0 to apply weight penalty, 0.0 to remove
theta=0.6

OUTPUT_FILE = 'NNResults_spect_seg.csv'
NUM_CHANNELS = 1
NUM_LABELS = 1

if UsePCA:
    NUM_FEATS=PCAfeats
else:
    NUM_FEATS=70
#DATA_FILE='Split Data_Standard_12-09-2016_auto_Normal_Abnormal'
if balanced:
    DATA_FILE='Split Data_Standard-&-Specto_12-09-2016_auto_Normal_Abnormal'
else:
    DATA_FILE='Split Data_Standard-&-Specto_12-09-2016_auto_Normal_Abnormal_unbalanced'


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
            # Input data
            tf_train_batch = tf.placeholder(
                tf.float64, shape=(batch_size, NUM_FEATS))
            #tf_train_batch = tf.placeholder(
            #    tf.float64, shape=(batch_size, NUM_FEATS))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, NUM_LABELS))
#            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size))
            tf_valid_dataset = tf.constant(self.val_X)
            #tf_test_dataset = tf.placeholder(
            #    tf.float64, shape=[len(self.test_X), NUM_FEATS])
            tf_test_dataset = tf.constant(self.test_X)
            tf_train_dataset = tf.placeholder(
                tf.float64, shape=[len(self.train_X), NUM_FEATS])

            # Implement dropout
            dropout_keep_prob = tf.placeholder(tf.float32)

            # Network weights/parameters that will be learned


            layer0a_weights = tf.Variable(tf.truncated_normal(
                [layer0_size[0], layer0_depth[0]], stddev=0.1))
            layer0a_biases = tf.Variable(tf.zeros([layer0_depth[0]]))

            layer0b_weights = tf.Variable(tf.truncated_normal(
                [layer0_size[1], layer0_depth[1]], stddev=0.1))
            layer0b_biases = tf.Variable(tf.zeros([layer0_depth[1]]))

            layer0c_weights = tf.Variable(tf.truncated_normal(
                [layer0_size[2], layer0_depth[2]], stddev=0.1))
            layer0c_biases = tf.Variable(tf.zeros([layer0_depth[2]]))

            layer0d_weights = tf.Variable(tf.truncated_normal(
                [layer0_size[3], layer0_depth[3]], stddev=0.1))
            layer0d_biases = tf.Variable(tf.zeros([layer0_depth[3]]))

            layer0e_weights = tf.Variable(tf.truncated_normal(
                [layer0_size[4], layer0_depth[4]], stddev=0.1))
            layer0e_biases = tf.Variable(tf.zeros([layer0_depth[4]]))

            layer0f_weights = tf.Variable(tf.truncated_normal(
                [layer0_size[5], layer0_depth[5]], stddev=0.1))
            layer0f_biases = tf.Variable(tf.zeros([layer0_depth[5]]))

            layer1_weights = tf.Variable(tf.truncated_normal(
                [np.sum(layer0_depth), layer1_depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([layer1_depth]))

            layer4_weights = tf.Variable(tf.truncated_normal(
                [layer1_depth, NUM_LABELS], stddev=0.1))
            layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

            # Model
            def network_model(data):

                '''Define the actual network architecture'''

                # Layer 1
                #conv1 = tf.nn.convolution(data, layer1_weights, [1, layer1_stride, layer1_stride, 1], padding='SAME')
                #hidden = tf.nn.relu(conv1 + layer1_biases)
                data=tf.cast(data, tf.float32)

                index1=0
                index2=layer0_size[0]
                hiddena = tf.nn.relu(tf.matmul(data[:,index1:index2], layer0a_weights) + layer0a_biases)
                index1=index2
                index2=index2+layer0_size[1]
                hiddenb = tf.nn.relu(tf.matmul(data[:,index1:index2], layer0b_weights) + layer0b_biases)
                index1=index2
                index2=index2+layer0_size[2]
                hiddenc = tf.nn.relu(tf.matmul(data[:,index1:index2], layer0c_weights) + layer0c_biases)
                index1=index2
                index2=index2+layer0_size[3]
                hiddend = tf.nn.relu(tf.matmul(data[:,index1:index2], layer0d_weights) + layer0d_biases)
                index1=index2
                index2=index2+layer0_size[4]
                hiddene = tf.nn.relu(tf.matmul(data[:,index1:index2], layer0e_weights) + layer0e_biases)
                index1=index2
                index2=index2+layer0_size[5]
                hiddenf = tf.nn.relu(tf.matmul(data[:,index1:index2], layer0f_weights) + layer0f_biases)

                index0 = 0
                index1 = layer0_size[0]
                index2 = index1 + layer0_depth[1]
                index3 = index2 + layer0_depth[2]
                index4 = index3 + layer0_depth[3]
                index5 = index4 + layer0_depth[4]
                index6 = index5 + layer0_depth[5]
                hidden = tf.nn.relu(tf.matmul(hiddena, layer1_weights[index0:index1,:]) +
                                    tf.matmul(hiddenb, layer1_weights[index1:index2,:]) +
                                    tf.matmul(hiddenc, layer1_weights[index2:index3,:]) +
                                    tf.matmul(hiddend, layer1_weights[index3:index4,:]) +
                                    tf.matmul(hiddene, layer1_weights[index4:index5,:]) +
                                    tf.matmul(hiddenf, layer1_weights[index5:index6,:]) +layer1_biases)

                output = tf.matmul(hidden, layer4_weights) + layer4_biases
                return output

            # Training computation
            logits = network_model(tf_train_batch)
            shape = tf_train_labels.get_shape().as_list()
            new_tf_labels=tf.reshape(tf_train_labels, [shape[0], 1])
            loss = tf.reduce_sum(abs(tf.nn.sigmoid(logits)-tf_train_labels))

            # Add weight decay penalty
            loss = loss + weight_decay_penalty([layer0a_weights, layer0b_weights, layer0c_weights,
                                                layer0d_weights, layer0e_weights, layer0f_weights,
                                                layer1_weights, layer4_weights], weight_penalty)

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
                            val_preds  = session.run(valid_prediction, feed_dict={dropout_keep_prob: 1.0})
                            test_preds = session.run(test_prediction, feed_dict={dropout_keep_prob: 1.0})
                            print('')
                            print('Batch loss at step %d: %f' % (step, l))
                            print('Batch training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                            print('Validation accuracy: %.1f%%' % accuracy(val_preds, self.val_Y))
                            print('Full train accuracy: %.1f%%' % accuracy(train_preds, self.train_Y))
                            #if accuracy(val_preds, self.val_Y)>70:
                            #    break
                            print('Test accuracy: %.1f%%' % accuracy(test_preds, self.test_Y))





                    # for theta in threshold:
                    n = len(self.val_Y)
                    matrix = np.zeros((n, 5))
                    matrix[:, 0] = np.array(self.val_Y).reshape(n) * 2 - 1
                    matrix[:, 1] = np.array(self.val_noise).reshape(n)

                    counter = 0
                    for i in val_preds:
                        if i > (0.5 + theta / 2):
                            matrix[counter, 2:5] = [1, 0, 0]
                        elif i < (0.5 - theta / 2):
                            matrix[counter, 2:5] = [0, 0, 1]
                        else:
                            matrix[counter, 2:5] = [0, 1, 0]

                        counter += 1

                    Sensitivity, Specificity, MAcc = DataSet.heart_sound_scoring(matrix)
                    # res[0, num], res[1, num], res[2, num] = DataSet.heart_sound_scoring(matrix)
                    # print('Final Results:')
                    print('Sensitivity: ', Sensitivity)
                    print('Specificity: ', Specificity)
                    print('MAcc: ', MAcc)
                    counter = 0
                    n = len(self.test_Y)
                    matrix = np.zeros((n, 5))
                    matrix[:, 0] = np.array(self.test_Y).reshape(n) * 2 - 1
                    matrix[:, 1] = np.array(self.test_noise).reshape(n)
                    for i in test_preds:
                        if i > (0.5 + theta / 2):
                            matrix[counter, 2:5] = [1, 0, 0]
                        elif i < (0.5 - theta / 2):
                            matrix[counter, 2:5] = [0, 0, 1]
                        else:
                            matrix[counter, 2:5] = [0, 1, 0]

                        counter += 1

                    TestSensitivity, TestSpecificity, TestMAcc = DataSet.heart_sound_scoring(matrix)
                    # res[0, num], res[1, num], res[2, num] = DataSet.heart_sound_scoring(matrix)
                    print('Testing Results')
                    print('Sensitivity: ', TestSensitivity)
                    print('Specificity: ', TestSpecificity)
                    print('MAcc: ', TestMAcc)

                    print('Dropout Probability: %1f' % dropout_prob)
                    print('Weight Penalty: %1f' % weight_penalty)
                    print('Number of Steps: %d' % num_training_steps)

                    if Writetofile:
                        fd = open(OUTPUT_FILE, 'a')
                        fd.write('\n %1f, %d, %d, %d, %d,' % (
                        theta, layer1_depth, layer3_depth, num_training_steps, batch_size))
                        fd.write('%1f, %1f,' % (dropout_prob, weight_penalty))
                        fd.write('%1f%%, %1f%%, %1f%%,' % (
                        accuracy(train_preds, self.train_Y), accuracy(val_preds, self.val_Y),
                        accuracy(test_preds, self.test_Y)))
                        fd.write('%1f%%, %1f%%, %1f%%,' % (Sensitivity * 100, Specificity * 100, MAcc * 100))
                        fd.write(
                            '%1f%%, %1f%%, %1f%%,' % (TestSensitivity * 100, TestSpecificity * 100, TestMAcc * 100))
                        fd.write('%r,' % balanced)
                        # fd.write('%r, %r, %r, %1f, %1f, %d, %.1f%%, %.1f%%' % (pooling, Augment, invariance, dropout_prob, weight_penalty, num_training_steps, accuracy(val_preds, self.val_Y), accuracy(train_preds, self.train_Y)))
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
        data = DataSet.load_data_set(pickle_file)
        data_type = 'auto'
        y_label = 'Normal/Abnormal'  # 'Normal/Abnormal'

        ''' Logistic Regression with Auto Data'''
        # Data Preparation
        self.test_noise = data[1]
        self.train_noise = data[4]
        self.val_noise = data[7]


        self.test_Y = data[0]/2+0.5
        self.train_Y = data[3]/2+0.5
        self.val_Y = data[6]/2+0.5

        Ydim=np.array(self.test_Y).shape
        self.test_Y=np.array(self.test_Y).reshape(Ydim[0],1)

        Ydim=np.array(self.train_Y).shape
        self.train_Y=np.array(self.train_Y).reshape(Ydim[0],1)

        Ydim=np.array(self.val_Y).shape
        self.val_Y=np.array(self.val_Y).reshape(Ydim[0],1)

        self.test_X = data[2]
        self.train_X = data[5]
        self.val_X = data[8]


        if UsePCA:

            pca = PCA(n_components=PCAfeats)
            pca.fit(self.train_X)
            self.train_X = pca.transform(self.train_X)
            self.val_X = pca.transform(self.val_X)
            self.test_X = pca.transform(self.test_X)

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
#                        dropout_prob=i/float(20)
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
#         for j in range(16,64,16):
#             layer1_depth=j
# #                    layer2_pool_stride=j
# #            layer2_filter_size=j
#             for k in range(16,64,16):
#                 layer3_depth=k
#                 for p in range(0,4,1):
#                         theta=float(p)/5
# #                    Augment=True
# #                    num_training_steps=6001
#                     for q in range (500,4500,500):
#                         num_training_steps=q
#                         print('%r %d %d %d %d' % (i,j,k,p,q))
# #                    print('%d %d' % (i,j))
        for j in range(16,64,16):
            layer1_depth=j
# #                    layer2_pool_stride=j
# #            layer2_filter_size=j
            for k in range(16,64,16):
                layer3_depth=k
                for p in range(0,4,1):
                        theta=float(p)/5
                        t1 = time.time()
                        conv_net = PCGNet()
                        conv_net.train_model()
                        t2 = time.time()
                        print ("Finished training. Total time taken:", t2 - t1)
#                    print "Total time elapsed:", t2 - tstart