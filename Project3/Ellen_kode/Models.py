import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree, datasets, preprocessing
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
#from skopt import BayesSearchCV
#from skopt.space import Real, Categorical, Integer
from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_cumulative_gain
from sklearn.metrics import RocCurveDisplay, multilabel_confusion_matrix, balanced_accuracy_score

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense           
from keras import optimizers             
from keras import regularizers


np.random.seed(0)

####################
#   Desicion tree
####################
"""
clf_base = tree.DecisionTreeClassifier(max_depth = i, class_weight ='balanced', 
                                       splitter ='best', min_samples_leaf = 1)
clf = AdaBoostClassifier(base_estimator = clf_base)

"""




####################################
#      Random Forest 
####################################
def run_RandomForest(X_train_scaled, y_train, X_test_scaled, y_test, n_est, eta, depth):

    rfc = RandomForestClassifier(n_estimators = n_est, max_depth = depth, 
                                 random_state = 0)
    
    # Train the classifier on the training data
    rfc.fit(X_train_scaled, y_train)

    # Make predictions on the test data
    y_pred_train = rfc.predict(X_train_scaled)
    y_pred_test = rfc.predict(X_test_scaled)

    # Evaluate the accuracy of the model
    Train_accuracy = accuracy_score(y_train, y_pred_train)
    Test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print("N Estimators: ", int(n_est), " Learning Rate: ", eta)
    print(f"Train Accuracy: {Train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {Test_accuracy * 100:.2f}%")  
    print(" ")

    return rfc, Train_accuracy, Test_accuracy





####################################
#      GBoost 
####################################
def run_GradientBoost(X_train_scaled, y_train, X_test_scaled, y_test, n_est, eta, depth):
    gbc = GradientBoostingClassifier(n_estimators = n_est, learning_rate = eta, 
                                     max_depth = depth, random_state = 0, verbose=1)
    print("Training GBoost model")
    # Train the classifier on the training data
    gbc.fit(X_train_scaled, y_train)
    
    print("Predict GBoost model")
    # Make predictions on the test data
    y_pred_train = gbc.predict(X_train_scaled)
    y_pred_test = gbc.predict(X_test_scaled)
    
    print("Evaluate GBoost model accuracy")
    # Evaluate the accuracy of the model
    Train_accuracy = accuracy_score(y_train, y_pred_train)
    Test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"GBoost Train Accuracy: {Train_accuracy * 100:.2f}%")
    print(f"GBoost Test Accuracy: {Test_accuracy * 100:.2f}%")
    
    return gbc, Train_accuracy, Test_accuracy







def plotTree(rfc, Acc, n_est, eta, depth):
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=1600)
    tree.plot_tree(rfc.estimators_[0])
    plt.title("Random Forrest, N_Estimators: " + str(n_est) + 
              "   Learning rate: " + str(eta) + 
              "   Depth: " + str(depth) + 
              "   Accuracy: " + str(round(Acc,3)))
    plt.show()
    return 
    
    
def plotGrid(data, x_ax, y_ax, T1):
    sns.set()
    fig, ax1 = plt.subplots(figsize = (10, 10))
    sns.heatmap(data, annot=True, ax=ax1, cmap="viridis",  fmt=".2%")
    ax1.set_title(T1)
    ax1.set_xlabel(x_ax)
    ax1.set_ylabel(y_ax)
    return
    
    
def plotMultiConfusion(y_test, y_pred,T1):
    conf_mat = multilabel_confusion_matrix(y_test, y_pred)/len(y_test)*100
    
    f, axes = plt.subplots(3, 4, figsize=(25, 15))
    axes = axes.ravel()
    for i in range(12):
        disp = ConfusionMatrixDisplay(conf_mat[i][0:2][0:2])
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(f'class {i}')
        plt.tick_params(axis=u'both', which=u'both',length=0)
        plt.grid(b=None)
        if i<8:
            disp.ax_.set_xlabel('')
        if i%4!=0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()
    
    plt.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.show()
    return
    

def plotConfusion(y_test, y_pred,T1):
    cm = confusion_matrix(y_test, y_pred, normalize='all')
    disp = ConfusionMatrixDisplay(cm, display_labels=['Hazardious','Non-hazardious'])
    disp.plot()
    #plot_confusion_matrix(y_test, y_pred)
    plt.tick_params(axis=u'both', which=u'both',length=0)
    plt.grid(b=None)
    plt.title(T1)
    plt.show()
    
    return


def plotROC (y_test, y_prob, T1):
    plt.figure()
    plot_roc(y_test, y_prob)
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(T1)
    plt.legend(fontsize='xx-small')
    plt.show()

    return        


def plotGain(y_true, y_prob,T1):
    plt.figure()
    plot_cumulative_gain(y_true, y_prob)
    plt.axis("square")
    plt.xlabel("Gain")
    plt.ylabel("Percentage of sample")
    plt.title(T1)
    plt.legend(fontsize='xx-small')
    plt.show()
