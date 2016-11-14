# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 17:12:22 2016

@author: Daniel Chamberlain
"""

# What functions do we need:
# Function to find best parameter values and return them
# Function to train a classifier based on the type

from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing
import sklearn.decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, KFold
from numpy.random import rand
import numpy as np
import scipy
from operator import itemgetter
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from scipy import interp

def train_classifier(x_train,y_train,
                     clf_type='lr',
                     lr_regularization = 'l1',
                     svc_kernel = 'rbf',
                     optimize_params = True,
                     use_pca = False,
                     param_optimization_iter = 100,
                     verbose = 0):
    
    # Define classifiers
    if clf_type == 'lr':
        clf = LogisticRegression(penalty = lr_regularization)
        param_dist = {"clf__C": scipy.stats.expon(scale=100)}
        has_prob = True
    elif clf_type == 'svc':
        clf = SVC(kernel=svc_kernel)
        param_dist = {'clf__C': scipy.stats.expon(scale=100), 
                      'clf__gamma': scipy.stats.expon(scale=.1)}                      
        has_prob = False
    elif clf_type == 'rf':
        clf = RandomForestClassifier(n_estimators=20)
        param_dist = {"clf__max_depth": [3, None],
              "clf__max_features": scipy.stats.randint(1, 11),
              "clf__min_samples_split": scipy.stats.randint(1, 11),
              "clf__min_samples_leaf": scipy.stats.randint(1, 11),
              "clf__bootstrap": [True, False],
              "clf__criterion": ["gini", "entropy"]}
        has_prob = True

    else:
        print('Classifier type {} not found'.format(clf_type))
        return -1
    
    if use_pca:
        clf = Pipeline([('scale', sklearn.preprocessing.StandardScaler()),
                        ('pca', sklearn.decomposition.PCA(0.95)),
                        ('clf', clf)])
    else:
        clf = Pipeline([('scale', sklearn.preprocessing.StandardScaler()),
                        ('clf', clf)])
    # Run parameter optimization over training set
    if optimize_params:
        random_search = RandomizedSearchCV(clf,param_distributions=param_dist,
                                           n_iter=param_optimization_iter,
                                           scoring = 'roc_auc',
                                           verbose=verbose)
        random_search.fit(x_train,y_train)
        if verbose > 0:        
            report(random_search.grid_scores_)
        params = random_search.best_params_
        clf.set_params(**params)
        
    # Train final model
    clf.fit(x_train,y_train)
    return clf,has_prob
    
def evaluate_classifier(clf,x_test,y_test,has_prob = True):
    if has_prob:
        y_score = clf.predict_log_proba(x_test)[:,1]
    else:
        y_score = clf.decision_function(x_test)
    auc = roc_auc_score(y_test,y_score)
    if auc < 0.5:
        auc = 1-auc
    return auc


def predict(clf,x,has_prob=True):
    if has_prob:
        y_score = clf.predict_proba(x)[:,1]
    else:
        y_score = clf.decision_function(x)
    return y_score
        
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")      
  
def test_train_classifier(n = 300, coeff=np.asarray([2,2]),noise_level = 2,
                                                    percent_test = 0.2):
    # Create randomized dataset
    x = rand(n,2)-0.5
    y = np.sign(np.dot(x,coeff) + noise_level*(rand(n)-0.5))
    x = np.column_stack((x,rand(n)-0.5))
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                     test_size=percent_test,
                                                     stratify=y)      
    
    clf = train_classifier(x_train,y_train,clf_type='lr',verbose=1)
    print('Test set auc: {:0.03f}'.format(
        evaluate_classifier(clf,x_test,y_test)))

def plot_feature_importance(varnames,coef):
    # Remove non-zero predictors
    nonzero = np.abs(coef) > 0
    coef_nonzero = coef[nonzero]
    varnames_nonzero = varnames[nonzero]
    coef_order = np.argsort(np.abs(coef_nonzero))
    print('Smallest to largest:')
    print(varnames_nonzero[coef_order])
    plt.barh(np.arange(len(coef_nonzero)),coef_nonzero[coef_order])
    plt.yticks(np.arange(len(coef_nonzero)), varnames_nonzero[coef_order])
    plt.tight_layout()
#    plt.show()
    

def plot_roc(clf,x_test,y_test,has_prob = True,fix_concave = True):
    if has_prob:
        y_score = clf.predict_log_proba(x_test)[:,1]
    else:
        y_score = clf.decision_function(x_test)
    
    # Compute basic AUC and ROC curve
    auc = roc_auc_score(y_test,y_score)
    fpr,tpr,thresholds = sklearn.metrics.roc_curve(y_test, y_score)
    # If probabilities are being given for the wrong class, flip them
    if auc < 0.5:
        fpr = 1-fpr
        tpr = 1-tpr
    
    # If we are fixing it to make it concave, check each combination of points 
    # and eliminate those below them
    if fix_concave:
        fpr,tpr = fix_roc_nonconcave(fpr,tpr)
    
    plt.plot(fpr,tpr,lw=3)
    plt.plot([0,1],[0,1],'k--',lw=3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    auc = sklearn.metrics.auc(fpr, tpr)
    plt.title('AUC: {:0.2f}'.format(auc))


def fix_roc_nonconcave(fpr,tpr):
    new_fpr = fpr
    new_tpr = tpr
    for m in np.arange(len(fpr)):
        for n in np.arange(m+2,len(fpr)):
            tpr_line = interp(fpr[m:n], np.asarray([fpr[m], fpr[n]]), np.asarray([tpr[m], tpr[n]]))
            for o in np.arange(len(tpr_line)):
                if tpr_line[o] > tpr[m+o]:
                    tpr[m+o] = fpr[m+o]-0.01
    non_concave_points = tpr < fpr
    fpr = new_fpr[np.logical_not(non_concave_points)]
    tpr = new_tpr[np.logical_not(non_concave_points)]
    return fpr,tpr


def generate_average_roc(x,y,classifier,has_prob,n_trials = 100,verbose=0):
    fpr = np.linspace(0, 1, 101)
    tpr = np.zeros((n_trials,len(fpr)))

    for m in np.arange(n_trials):
        kf = sklearn.cross_validation.StratifiedKFold(y,n_folds = 5,shuffle=True)
        for train_index, test_index in kf:
            x_train, x_test = x[train_index,:], x[test_index,:]
            y_train, y_test = y[train_index], y[test_index]

            classifier.fit(x_train,y_train)
            y_prob = predict(classifier,x_test,has_prob=has_prob)
            fpr_trial, tpr_trial, thresholds = sklearn.metrics.roc_curve(y_test, y_prob,pos_label = True)
            tpr[m,:] = interp(fpr, fpr_trial, tpr_trial)
            tpr[m,0] = 0
            tpr[m,-1] = 1

            if verbose>0:
                print('Fold AUC: {:0.2f}'.format(sklearn.metrics.roc_auc_score(y_test,y_prob)))
                
    median_tpr = np.median(tpr,axis=0)
    low_tpr = np.percentile(tpr,5,axis=0)
    high_tpr = np.percentile(tpr,95,axis=0)
    
    [median_fpr,median_tpr] = fix_roc_nonconcave(fpr,median_tpr)
    [low_fpr,low_tpr] = fix_roc_nonconcave(fpr,low_tpr)
    [high_fpr,high_tpr] = fix_roc_nonconcave(fpr,high_tpr)
    plt.plot(median_fpr,median_tpr)
    plt.plot(low_fpr,low_tpr)
    plt.plot(high_fpr,high_tpr)
#    plt.show()
#    mean_tpr = mean_tpr/(len(kf)*n_trials)
#    mean_tpr[0] = 0
#    mean_tpr[-1] = 1

    # Fix non-concave portions
    
#    plt.plot(fpr,tpr)
    median_auc = sklearn.metrics.auc(median_fpr, median_tpr)
    high_auc = sklearn.metrics.auc(high_fpr, high_tpr)
    low_auc = sklearn.metrics.auc(low_fpr, low_tpr)
    plt.title('AUC = {:0.02f}'.format(median_auc))
#    plt.show()
    return median_auc,low_auc,high_auc


if __name__ == '__main__':
    test_train_classifier()