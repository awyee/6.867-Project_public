import tensorflow as tf
import numpy as np
from DataSet import DataSet
from six.moves import cPickle as pickle
import sys
import math
import time
from sklearn.decomposition import PCA

UsePCA=True
Writetofile=True

# Hyperparameters
batch_size = 200
learning_rate = 0.01
num_training_steps = 3401
gamma=float(-2)

theta=0.6

C=1

OUTPUT_FILE = 'SVMResults_spect.csv'
NUM_CHANNELS = 1
NUM_LABELS = 1
PCAFeats=20
if  UsePCA:
    NUM_FEATS=PCAFeats
else:
    NUM_FEATS=70
#DATA_FILE='Split Data_Standard_12-09-2016_auto_Normal_Abnormal'
DATA_FILE='Split Data_Standard-&-Specto_12-09-2016_auto_Normal_Abnormal'


class PCG_SVM:
    def __init__(self):
        '''Initialize the class by loading the required datasets
		and building the graph'''
        self.load_pickled_dataset(DATA_FILE)
        self.graph=tf.Graph()
        self.define_SVM()

    def define_SVM(self):
        print ('\nDefining model...')

        with self.graph.as_default():
            # Input data
            tf_train_batch = tf.placeholder(shape=[batch_size, NUM_FEATS], dtype=tf.float32)
            tf_valid_dataset = tf.constant(self.val_X)
            tf_test_dataset = tf.constant(self.test_X)
            tf_train_dataset = tf.placeholder(
                tf.float64, shape=[len(self.train_X), NUM_FEATS])
            tf_train_labels = tf.placeholder(shape=[batch_size, NUM_LABELS], dtype=tf.float32)

            b = tf.Variable(tf.random_normal(shape=[1, batch_size]))


            # Model

            dist = tf.reduce_sum(tf.square(tf_train_batch), 1)
            dist = tf.reshape(dist, [-1, 1])
            sq_dists = tf.add(tf.sub(dist, tf.mul(2., tf.matmul(tf_train_batch, tf.transpose(tf_train_batch)))),
                                  tf.transpose(dist))
            kernel = tf.exp(tf.mul(gamma, tf.abs(sq_dists)))

            model_output = tf.transpose(tf.matmul(b, kernel))
            first_term = tf.reduce_sum(b)
            b_vec_cross = tf.matmul(tf.transpose(b), b)
            y_target_cross = tf.matmul(tf_train_labels, tf.transpose(tf_train_labels))
            second_term = tf.reduce_sum(tf.mul(kernel, tf.mul(b_vec_cross, y_target_cross)))
            loss = tf.neg(tf.sub(first_term, second_term))-10000*tf.minimum( tf.reduce_min(C-b),0)

            # Gaussian (RBF) prediction kernel
            def GaussRBF(data):
                rA = tf.reshape(tf.reduce_sum(tf.square(tf_train_batch), 1), [-1, 1])
                rB = tf.reshape(tf.reduce_sum(tf.square(data), 1), [-1, 1])
                rB = tf.cast(rB, tf.float32)
                data = tf.cast(data, tf.float32)
                pred_sq_dist = tf.add(tf.sub(rA, tf.mul(2., tf.matmul(tf_train_batch, tf.transpose(data)))),
                                      tf.transpose(rB))
                pred_kernel = tf.exp(tf.mul(gamma, tf.abs(pred_sq_dist)))

                prediction_output = tf.matmul(tf.mul(tf.transpose(tf_train_labels), b), pred_kernel)
                # prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
                # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))
                prediction_output=tf.transpose(prediction_output)
                return prediction_output


            # Training computation

            # Optimizer
            my_opt = tf.train.GradientDescentOptimizer(0.002)
            train_step = my_opt.minimize(loss)

            # Predictions for the training, validation, and test data.
            batch_prediction = model_output
            valid_prediction = GaussRBF(tf_valid_dataset)
            test_prediction = GaussRBF(tf_test_dataset)
            train_prediction = GaussRBF(tf_train_dataset)

            def train_model(num_steps=num_training_steps):
                '''Train the model with minibatches in a tensorflow session'''
                with tf.Session(graph=self.graph) as session:
                    tf.initialize_all_variables().run()
                    print ('Initializing variables...')

                    for step in range(num_steps):
                        # offset = (step * batch_size+startloc) % (self.train_Y.shape[0] - batch_size)
                        # batch_data = self.train_X[offset:(offset + batch_size), :]
                        # batch_labels = self.train_Y[offset:(offset + batch_size)]
                        #
                        # # Data to feed into the placeholder variables in the tensorflow graph
                        # feed_dict = {tf_train_batch: batch_data, tf_train_labels: batch_labels}

                        rand_index = np.random.choice(len(self.train_Y), size=batch_size)
                        batch_data = self.train_X[rand_index,:]
                        batch_labels = self.train_Y[rand_index]
                        feed_dict={tf_train_batch: batch_data, tf_train_labels: batch_labels}

                        _, l, predictions = session.run([train_step, loss, batch_prediction], feed_dict=feed_dict)
                        if (step % 200 == 0):
                            train_preds = session.run(train_prediction, feed_dict={tf_train_dataset: self.train_X,
                                                                                   tf_train_batch: batch_data,
                                                                                   tf_train_labels: batch_labels})
                            val_preds  = session.run(valid_prediction, feed_dict=feed_dict)
                            test_preds = session.run(test_prediction, feed_dict=feed_dict)
                            print('')
                            print('Batch loss at step %d: %f' % (step, l))
                            print('Batch training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                            print('Validation accuracy: %.1f%%' % accuracy(val_preds, self.val_Y))
                            print('Full train accuracy: %.1f%%' % accuracy(train_preds, self.train_Y))
                            #if accuracy(val_preds, self.val_Y)>70:
                            #    break
                            print('Test accuracy: %.1f%%' % accuracy(test_preds, self.test_Y))


                    n = len(self.val_Y)
                    matrix = np.zeros((n, 5))
                    matrix[:, 0] = np.array(self.val_Y).reshape(n)
                    matrix[:, 1] = np.array(self.val_noise).reshape(n)

                    counter = 0
                    for i in val_preds:
                        if np.sign(i)>theta:
                            matrix[counter, 2:5] = [1, 0, 0]
                        elif np.sign(i)<-theta:
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

                    print('')
                    print('Parameters')
                    print('Number of Steps: %d' % num_training_steps)




                    if Writetofile:
                        fd = open(OUTPUT_FILE, 'a')
                        fd.write('\n %1f, %d, %d, %1f, %1f, %r,' % (theta, num_training_steps, batch_size, C, gamma, UsePCA))
                        fd.write('%1f%%, %1f%%, %1f%%,' % (accuracy(train_preds, self.train_Y),
                                                           accuracy(val_preds, self.val_Y),
                                                           accuracy(test_preds, self.test_Y)))
                        fd.write('%1f%%, %1f%%, %1f%%,' % (Sensitivity*100, Specificity*100,MAcc*100))
                        #fd.write('%r, %r, %r, %1f, %1f, %d, %.1f%%, %.1f%%' % (pooling, Augment, invariance, dropout_prob, weight_penalty, num_training_steps, accuracy(val_preds, self.val_Y), accuracy(train_preds, self.train_Y)))
                        fd.close()

            self.train_model = train_model

    def load_pickled_dataset(self, pickle_file):

        data = DataSet.load_data_set(pickle_file)
        data_type = 'auto'
        y_label = 'Normal/Abnormal'  # 'Normal/Abnormal'

        ''' Logistic Regression with Auto Data'''
        # Data Preparation
        self.test_noise = data[1]
        self.train_noise = data[4]
        self.val_noise = data[7]

        self.test_Y = data[0]
        self.train_Y = data[3]
        self.val_Y = data[6]

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

            pca = PCA(n_components=PCAFeats)
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
    classifications= np.sign(predictions)
    return (100.0 * sum(classifications == labels)/ predictions.shape[0])
#    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
#            / predictions.shape[0])


if __name__ == '__main__':
        # for j in range(0,2,1):
        #     C=pow(10,float(j)/2)
        #     for k in range(-4,1,1):
        #         gamma=-pow(10,float(k)/2)
        #         for p in range(0,8,2):
        #                 theta=float(p)/10
        # for j in range (100, 1001, 100):
        #                 batch_size=j
                        t1 = time.time()
                        conv_net = PCG_SVM()
                        conv_net.train_model()
                        t2 = time.time()
                        print ("Finished training. Total time taken:", t2 - t1)
#                    print "Total time elapsed:", t2 - tstart
