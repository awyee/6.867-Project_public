# 7th November
# Implementation Niccolò
# From Levi

# NB the SGD implemented in this code uses a constant learning rate
# For better performance change the learning rate to a decreasing one
# This will optimise convergence
# E.g. n(i)= (1/(n(i-1)+i)^(-0.75)

from matplotlib import pyplot as plt
import numpy as np
import pickle
import sys
from collections import OrderedDict

# Variables
NETWORK = [2,2,2]
CONVERGENCE_CRITERION = 1e-4
ETA = 1e-5
# DATA_SET = 1 # For 2D data if b/w 1 and 4; '5' calls data_3class
N_ITERATIONS = 10000

# Visualisations
COLOUR_ONE = 'green'
COLOUR_TWO = 'orange'
TITLE = False
# FINAL_MESSAGE = 'DataSet_' + str(DATA_SET)


''' Functions '''


def reLU(x):

    '''
    ReLU. No Used.
    '''
    output = np.zeros(x.shape)
    for c,j in enumerate(x):
        output[c] = np.max([0,j])
    return output


def softmax(x):
    '''
    Softmax. Used.
    '''
    output = np.zeros(x.shape)
    #Note here, the alpha is to prevent overflow
    #Apparently it is common in softmax
    #The alphas is just a 'large' number to prevent that.
    #See: http://deeplearning.stanford.edu/wiki/index.php/Exercise:Softmax_Regression#Step_2:_Implement_softmaxCost
    #And: http://eric-yuan.me/softmax/
    alpha = np.max(x)
    t = np.sum(np.exp(x-alpha))
    for c,j in enumerate(x):
        output[c] = np.exp(x[c]-alpha)/t
    return output


def reLUP(x):
    '''
        ReLU derivative. Not used
    '''
    output = np.zeros(x.shape)
    for c,j in enumerate(x):
        output[c] = 1 if j > 0 else 0
    return output


def sigmoid(x):
    '''
    Sigmoid, if you want it. Not used currently
    '''
    output = np.zeros(x.shape)
    for c,j in enumerate(x):
        output[c] = 1./(1.+np.exp(-j))
    return output


def sigmoidP(x):
    '''
        The derivative of Sigmoid. Not used
    '''
    x = np.array(x)
    output = np.zeros(x.shape)
    for c,j in enumerate(x):
        s = sigmoid(np.array([j]))
        output[c] = s*(1.-s)
    return output

def load_network(filename='network.p'):
  '''
    This loads the network. Note this will not work if you have not imported LLN
  '''
  return pickle.load(open(filename,'rb'))


''' Learning Neural Net Class '''


class LNN:

    '''
        This class was created from a helpful guide at
        http://www.bogotobogo.com/python/python_Neural_Networks_Backpropagation_for_XOR_using_one_hidden_layer.php
    '''

    def __init__(self, layers, factivation=sigmoid, activation=reLU, niter=10000, eta=0.01, cc= 1e-4, Print=False):
        '''
            This mainly sets up the network. Keeping track of the weights (called layers in this implementation)
            is the hard part
        '''
        self.activation = activation
        self.niter = int(niter)
        self.learn = eta
        self.Print = Print
        self.factivation = factivation

        if self.factivation == sigmoid:
            self.factivationP = sigmoidP
        if self.activation == sigmoid:
            self.activationP = sigmoidP
        elif self.activation == reLU:
            self.activationP = reLUP

        self.weights= []
        self.actualiter = []
        self.cc= cc
        self.savename= ''

        # Save name with network architecture
        for i in layers:
            self.savename +=str(i) +'-'

        # Add eta to the name and remove last hyphen
        self.savename= FINAL_MESSAGE + '_' + self.savename[:-1] + '_' + 'Eta' + '_' + str(eta) + '_'

        # Initialise the weights
            # N.B. Weight matrices also include the bias term
            # We append ndarray in a weight list
        for i in range(1, len(layers) - 1):
            stddev = 1. / np.sqrt(layers[i - 1] + 1)
            r = np.random.normal(0, stddev, (layers[i - 1] + 1, layers[i] + 1))
            self.weights.append(r)
        stddev = 1. / np.sqrt(layers[i] + 1)
        r = np.random.normal(0, stddev, (layers[i] + 1, layers[i + 1]))
        self.weights.append(r)

    def feedforward(self,x):
        '''
          Fires the Network
        '''

        # Outputs become inputs
        self.output = [np.append(x, 1.0)]
        self.z= []

        for i in range(len(self.weights)-1):
            temp = np.dot(self.output[i], self.weights[i])
            self.z.append(temp)
            self.output.append(self.activation(temp))

        # Now perform the final activation
        temp= np.dot(self.output[i + 1], self.weights[i + 1])
        self.z.append(temp)
        self.output.append(self.factivation(temp))

        # Temporary return
        return self.output

    def backprop(self,x,y):
        '''
            The backprop algorithm. Slightly different than the diag stuff from class as I am storing the weights differently.
            Seems to work even for the softmax
        '''
        self.feedforward(x) # Puts result in self.output & self.z

        error = y - self.output[-1] # Looking at the last output of the net

        # This is the softmax error stuff
        # Taken from:
        # http://cs231n.github.io/neural-networks-case-study/#linear
        if self.factivation == softmax:
            # This is the error times gradient, taken from:
            # https://www.ics.uci.edu/~pjsadows/notes.pdf
            delta = [error]
        else:
            delta= [error*self.factivationP(self.output[-1])]

        for i in range(len(self.output) - 2, 0, -1):
            delta.append(delta[-1].dot(self.weights[i].T)*self.activationP(self.output[i]))
        delta.reverse()

        self.delta = delta
        self.error = error


    def terror(self, X, Y):
        '''
            Error from: https://www.ics.uci.edu/~pjsadows/notes.pdf
            This calculates the softmax error
        '''
        E = 0.0
        for c, i in enumerate(X):
            self.feedforward(i)
            t1 = np.dot(Y[c], np.log(self.output[-1]))
            E -= t1
        return E

    def train(self,train,validate,CC=0.0):

        # NB the SGD implemented in this code uses a constant learning rate
        # For better performance change the learning rate to a decreasing one
        # This will optimise convergence
        # E.g. n(i)= (1/(n(i-1)+i)^(-0.75)

        '''
            The actual training point.
            Run to niter || CC
        '''
        X = train[0]
        Y = train[1]
        vX = validate[0]
        vY = validate[1]

        pE = 1000
        for i in range(self.niter):
            t = np.random.permutation(X.shape[0]) # SGD
            for zz in t:
                data = X[zz]
                self.backprop(data,Y[zz]) # Get Output (i.e. a's) and Delta
                for c in range(len(self.weights)):
                    # View inputs as Arrays
                    layer = np.atleast_2d(self.output[c])
                    deltas= np.atleast_2d(self.delta[c])
                    self.weights[c] += self.learn * layer.T.dot(deltas) # TODO ask if * is efficient or need numpy
            if i % 10 ==0:
                verror = self.terror(vX, vY)
                terror = self.terror(X, Y)
                # f.write(str(i) + '\t' + str(terror) + '\t' + str(verror) + '\n')
                dE = np.abs(pE - verror)
                if dE <= CC:
                    print('Converged on Validation Data')
                    break
                pE = verror
            if i % 100 == 0:
                print('niter: ' + str(i))
                print('error: ' + str(self.terror(vX, vY)))

        self.actualiter= i

        f = open(self.savename + 'train' + '.out', 'w')
        f.write('iteration\tTrain Entropy Loss\tValidation Entropy Loss\n')
        f.write(str(i) + '\t' + str(terror) + '\t' + str(verror) + '\n')
        f.close()

        self.save_network(filename=self.savename + '.p')

    def save_network(self, filename='network.p'):
        '''
          This saves the state information so you can reload the network
        '''
        pickle.dump(self, open(filename, 'wb'))

    def getPred(self):
        return np.argmax(self.output[-1])

    def scorefn(self, x):
        self.feedforward(x)
        pred= self.getPred()
        if pred == 0:
            pred = -1
        return pred


def load_Data(filename):
  '''
    This function simply loads the data into X,Y and returns them as a tuple
  '''
  train = np.loadtxt('data/' + filename)
  X = train[:,0:2]
  Y = train[:,2:3]
  return (X,Y)


''' Data Sets '''


def load_SVM(dataset=1):
  train = 'data' + str(dataset) + '_train.csv'
  validate = 'data' + str(dataset) + '_validate.csv'
  test = 'data' + str(dataset) + '_test.csv'
  tX,tY = load_Data(train)
  vX,vY = load_Data(validate)
  testX,testY = load_Data(test)
  ctY = []
  cvY = []
  ctestY = []
  for i in tY:
    if i == -1:
      ctY.append([0,1])
    else:
      ctY.append([1,0])
  for i in vY:
    if i == -1:
      cvY.append([0,1])
    else:
      cvY.append([1,0])
  for i in testY:
    if i == -1:
      ctestY.append([0,1])
    else:
      ctestY.append([1,0])
  return (tX,ctY,vX,cvY,testX,ctestY)

def load_SVM3(dataset=5):

    filename= 'data_3class.csv'
    X,Y = load_Data(filename)

    init = int(np.floor(len(X) / 3))
    tX = X[0:init,:]
    tY = Y[0:init,:]
    vX = X[init:init*2,:]
    vY = Y[init:init*2,:]
    init = init * 2
    testX = X[init:init*2,:]
    testY = Y[init:init*2,:]

    ctY = []
    cvY = []
    ctestY = []

    for i in tY:
        if i == 0:
            ctY.append([1,0,0])
        elif i == 1:
            ctY.append([0,1,0])
        else:
            ctY.append([0,0,1])

    for i in vY:
        if i == 0:
            cvY.append([1,0,0])
        elif i == 1:
            cvY.append([0,1,0])
        else:
            cvY.append([0,0,1])

    for i in testY:
        if i == 0:
            ctestY.append([1,0,0])
        elif i == 1:
            ctestY.append([0,1,0])
        else:
            ctestY.append([0,0,1])

    return (tX,ctY,vX,cvY,testX,ctestY)

def load_MNIST(d1,d2,numtrain=200,normalize=False):
  '''
    This takes in the MNIST data
  '''
  tX = np.empty((0,784))
  tY = np.empty((0,2))
  vX = np.empty((0,784))
  vY = np.empty((0,2))
  for i in d1:
    x = np.loadtxt('data/mnist_digit_' + str(i) + '.csv')
    tX = np.vstack((tX,x[0:numtrain,:]))
    vX = np.vstack((vX,x[numtrain+1:numtrain+151,:]))
    for j in range(numtrain):
      tY = np.vstack((tY,[1,0]))
    for j in range(150):
      vY = np.vstack((vY,[1,0]))
  for i in d2:
    x = np.loadtxt('data/mnist_digit_' + str(i) + '.csv')
    tX = np.vstack((tX,x[0:numtrain,:]))
    vX = np.vstack((vX,x[numtrain+1:numtrain+151,:]))
    for j in range(numtrain):
      tY = np.vstack((tY,[0,1]))
    for j in range(150):
      vY = np.vstack((vY,[0,1]))
  if normalize:
    tX = 2.*tX/255. - 1.
    vX = 2.*vX/255. - 1.
  return (tX,tY,vX,vY)


def SVM(dataset=1,network=[2,8,2],niter=2000,eta=0.00001,CC=0.0):
    '''
    This does the SVM NN
    '''
    if(dataset==5):
        tX, tY, vX, vY, _, _ = load_SVM3(dataset=dataset)
    else:
        tX,tY,vX,vY,_,_ = load_SVM(dataset=dataset)

    train = [tX,tY]
    validate = [vX,vY]

    # savename = ''
    # for i in network:
    #     savename += str(i) + '-'
    # savename = savename[:-1]

    ''' for a single eta value '''
    # x = LNN(network,niter=niter,factivation=softmax,eta=eta)
    # f = open(x.savename + 'Errors.txt', 'w')
    # print('----- Eta is ' + str(eta) + ' ----')
    # x.train(train,validate,CC)
    # finalE= test_SVM(dataset=dataset,network=network,savename=x.savename)
    # f.write('eta\tTrain error\n')
    # f.write(str(eta) + '\t' + str(finalE[0]) + '\n')
    # f.write('eta\tTest error\n')
    # f.write(str(eta) + '\t' + str(finalE[1]) + '\n')

    ''' loop through eta values to train '''
    res_error = []
    etas = []
    aiter= []

    # for eta in np.power(10.0,range(-5,-2)):
    #     print('----- Eta is ' + str(eta) + ' ----')
    #     x = LNN(network,niter=niter,factivation=softmax,eta=eta,cc= CC)
    #     x.train(train,validate,CC)
    #     finalE = test_SVM(dataset,network,savename=x.savename)
    #     res_error.append(finalE)
    #     etas.append(eta)
    #     aiter.append(x.actualiter)

    print('----- Eta is ' + str(eta) + ' ----')
    x = LNN(network,niter=niter,factivation=softmax,eta=eta,cc= CC)
    x.train(train,validate,CC)
    finalE = test_SVM(dataset,network,savename=x.savename)
    res_error.append(finalE)
    etas.append(eta)
    aiter.append(x.actualiter)

    f = open(x.savename + 'Errors.txt', 'w')
    f.write('NumberIter:\t' + str(x.niter) + '\t' + 'CC:\t' + str(x.cc) + '\n')
    f.write('-----------------------------\n ')
    f.write('eta\tA_Iter\tE_Train\tE_Test\n')
    f.write('-----------------------------\n ')

    for i in range(0,(len(res_error))):
        f.write(str(etas[i]) + '\t' + str(aiter[i])+ '\t' + res_error[i][0] + '\t' +  res_error[i][1] + '\n')

    # f.write('Cross Entropy Validation Loss: ' + N_ITERATIONS + '\n')

    f.close()


def test_SVM(dataset=1, network=[2, 8, 2], savename=''):
    '''
        If ou specify the dataset number and the network used, it will load the network and then calculate the total error
    '''
    # if savename == '':
    #     for i in network:
    #         savename += str(i) + '-'
    #     savename = savename[:-1]

    x = load_network(savename + '.p')

    if (dataset == 5):
        tX, tY, vX, vY, testX, testY= load_SVM3(dataset=dataset)
    else:
        tX, tY, vX, vY, testX, testY = load_SVM(dataset=dataset)

    # Train Error
    error = 0.0
    total = 0.0
    allReal = []
    allPred = []
    #  print('target\tprediction')
    for c, i in enumerate(tX):
        x.feedforward(i)
        real = np.argmax(tY[c])
        predict = x.getPred()
        #    print(str(real)+'\t'+str(predict))
        if real != predict:
            error += 1
        total += 1
        allReal.append(real)
        allPred.append(predict)
    finalE = [str(100 * float(error) / float(total)) + '%']

    # Test Error
    error = 0.0
    total = 0.0
    allReal = []
    allPred = []
    #  print('target\tprediction')
    for c, i in enumerate(testX):
        x.feedforward(i)
        real = np.argmax(testY[c])
        predict = x.getPred()
        #    print(str(real)+'\t'+str(predict))
        if real != predict:
            error += 1
        total += 1
        allReal.append(real)
        allPred.append(predict)
    finalE.append(str(100 * float(error) / float(total)) + '%')

    print('Training Error: ' + finalE[0])
    print('Test Error: ' + finalE[1])

    # plotterComp(testX, allPred, allReal,savename)
    plotDecisionBoundary(testX, allReal, x, testX, title="")
    # plt.savefig(savename + '.jpeg', bbox_inches='tight', pad_inches=0)
    # plt.close("all")

    return finalE

def MNIST(d1=[1], d2=[2], numtrain=200, network=[784, 10, 2], niter=2000, eta=0.00001, normalize=True):
  tX, tY, vX, vY = load_MNIST(d1, d2, numtrain=numtrain, normalize=normalize)
  train = [tX, tY]
  validate = [vX, vY]
  x = LNN(network, niter=niter, factivation=softmax, eta=eta)
  x.train(train, validate)

def validate_MNIST(d1=[1], d2=[2], network=[784, 10, 2], numtrain=200, normalize=True):
  tX, tY, vX, vY = load_MNIST(d1, d2, numtrain=numtrain, normalize=normalize)
  savename = ''
  for i in network:
      savename += str(i) + '-'
  savename = savename[:-1]
  x = load_network(savename + '.p')
  error = 0.0
  total = 0.0
  print('target\tprecition')
  for c, i in enumerate(vX):
      x.feedforward(i)
      real = np.argmax(vY[c])
      predict = np.argmax(x.output[-1])
      print(str(real) + '\t' + str(predict))
      if real != predict:
          error += 1
      total += 1
  print('Error: ' + str(100 * float(error) / float(total)) + '%')


''' Visualisation Code '''

def plotter(X, Y):
    '''
      This function simply creates a scatter plot of the (X0,X1,+-1) data
    '''
    for c, i in enumerate(X):
        color = COLOUR_ONE if Y[c] == 0 else COLOUR_TWO
        label = '-1' if Y[c] == 0 else '+1'
        plt.scatter(i[0], i[1], color=color, label=label)


def plotterComp(X, Y, yval, savename):
    '''
      This function simply creates a scatter plot of the (X0,X1,+-1) data
    '''
    for c, i in enumerate(X):
        if Y[c] == 0 and yval[c] == 0:
            color = COLOUR_TWO
            label = 'Class 2'
        elif Y[c] == 0 and yval[c] == 1:
            color = 'magenta'
            label = 'Misclassified as Class 2'
        elif Y[c] == 1 and yval[c] == 0:
            color = 'cyan'
            label = 'Misclassified as Class 1'
        else:
            color = COLOUR_ONE
            label = 'Class 1'
        plt.scatter(i[0], i[1], color=color, label=label,  marker='x')
    x0min, x0max = X[:, 0].min() - 1, X[:, 0].max() + 2
    x1min, x1max = X[:, 1].min() - 1, X[:, 1].max() + 2
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    print('---Creating plot---')
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.xlabel('x$_0$', fontsize=16)
    plt.ylabel('x$_1$', fontsize=16)
    if TITLE: plt.title('NN Dataset ' + str(DATA_SET), fontsize=24)
    plt.xlim([x0min, x0max])
    plt.ylim([x1min, x1max])
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(savename + '.png', bbox_inches='tight', pad_inches=0.03)
    # plt.show()

def makePlots(x, y):
    '''
      This function creates plots of the data and the predictions
    '''
    plotter(x, y)
    x0min, x0max = x[:, 0].min() - 1, x[:, 0].max()
    x1min, x1max = x[:, 1].min() - 1, x[:, 1].max()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    print('---Displaying the plot---')
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel('x$_0$', fontsize=16)
    plt.ylabel('x$_1$', fontsize=16)
    if TITLE: plt.title('NN Dataset-' + str(DATA_SET), fontsize=24)
    plt.xlim([x0min, x0max])
    plt.ylim([x1min, x1max])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()

def plotDecisionBoundary(X, Y, net, values, title = ""):

    # X is data matrix (each row is a data point)
    # Y is desired output (1 or -1)
    # scoreFn is a function of a data point
    # values is a list of values to plot
    # Plot the decision boundary. For that, we will asign a score to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].

    # Convert Y
    Y = np.array(Y)
    values= np.array(values)
    Y[Y<1] = -1

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
    zz = np.array([net.scorefn(x) for x in np.c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    plt.figure()
    CS = plt.contour(xx, yy, zz, [-1,0,1], colors = 'green', linestyles = 'solid', linewidths = 2)
    plt.clabel(CS, fontsize=9, inline=1)
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = plt.cm.cool)
    plt.title(title)
    plt.axis('tight')
    plt.savefig(net.savename + '.png', bbox_inches='tight', pad_inches=0.03)


def message(set,net):
    print('\n______________________________________________\n')
    print('__' + FINAL_MESSAGE + '__' + 'Network: ' + str(net))
    print('\n______________________________________________\n')

''' Running Script '''
# x = LNN([2,10,2],niter=10000,factivation=softmax,eta=0.00001)
'''
  To run SVM:
  Train: SVM(dataset=1,network=[2,8,2],niter=2000,eta=0.00001)
  Validate: validate_SVM(dataset=1,network=[2,8,2])
  network saved to 2-8-2.p and training output saved to 2-8-2.out
  Uncomment:
'''

# AGGIUNGI 2,1,2
# Networks= [[2,2,2],[2,5,2],[2,10,2],[2,20,2],[2,2,2,2],[2,5,5,2],[2,10,10,2],[2,20,20,2]]
# Datasets= [2,3,4]
# N_ITERATIONS= 10000
# FINAL_MESSAGE = 'DataSet_' + str(set)
#
# for set in Datasets:
#     for net in Networks:
#         message(set, net)
#         SVM(set, net, N_ITERATIONS, ETA, CONVERGENCE_CRITERION)


# Optimised Data Set 2

set= 2
FINAL_MESSAGE = 'DataSet_' + str(set)

# net = [2,20,20,2]
# ETA= 1e-6
# message(set, net)
# SVM(set, net, N_ITERATIONS, ETA, CONVERGENCE_CRITERION)

net = [2,10,10,2]
ETA= 1e-5
message(set, net)
SVM(set, net, N_ITERATIONS, ETA, CONVERGENCE_CRITERION)


#
# set= 3
# FINAL_MESSAGE = 'DataSet_' + str(set)
#
# net = [2,5,2]
# ETA= 1e-5
# message(set, net)
# SVM(set, net, N_ITERATIONS, ETA, CONVERGENCE_CRITERION)
#
# net = [2,5,5,2]
# ETA= 1e-5
# message(set, net)
# SVM(set, net, N_ITERATIONS, ETA, CONVERGENCE_CRITERION)
#
# net = [2,20,2]
# ETA= 1e-5
# message(set, net)
# SVM(set, net, N_ITERATIONS, ETA, CONVERGENCE_CRITERION)
#
# net = [2,20,20,2]
# ETA= 1e-5
# message(set, net)
# SVM(set, net, N_ITERATIONS, ETA, CONVERGENCE_CRITERION)
#
#
# set= 4
# FINAL_MESSAGE = 'DataSet_' + str(set)
#
# net = [2,5,2]
# ETA= 0.001
# message(set, net)
# SVM(set, net, N_ITERATIONS, ETA, CONVERGENCE_CRITERION)
#
# net = [2,5,5,2]
# ETA= 1e-5
# message(set, net)
# SVM(set, net, N_ITERATIONS, ETA, CONVERGENCE_CRITERION)
#
# net = [2,10,2]
# ETA= 0.001
# message(set, net)
# SVM(set, net, N_ITERATIONS, ETA, CONVERGENCE_CRITERION)
#
# net = [2,20,2]
# ETA= 0.0001
# message(set, net)
# SVM(set, net, N_ITERATIONS, ETA, CONVERGENCE_CRITERION)
#
# net = [2,20,20,2]
# ETA= 0.0001
# message(set, net)
# SVM(set, net, N_ITERATIONS, ETA, CONVERGENCE_CRITERION)


'''
  To run MNIST:
  Train: MNIST(d1=[1],d2=[2],numtrain=200,network=[784,10,2],niter=2000,eta=0.00001,normalize=True):
  d1,d2 can be lists. I.E. To train even v odd: d1 = [,1,3,5,7,9], d2 = [0,2,4,6,8]
  Validate: validate_MNIST(d1=[1],d2=[2],numtrain=200,network=[784,10,2],normalize=True)
  note: d1,d2,numtrain,normalize and network need to be the same between MNIST and validate
  network saved to 784-10-2.p and training output saved to 784-10-2.out
  uncomment:
'''

#MNIST(d1=[1,3,5,7,9],d2=[0,2,4,6,8],numtrain=100,network=[784,10,2],niter=1000,eta=0.00001,normalize=True)
#validate_MNI