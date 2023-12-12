import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree, datasets, preprocessing
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
#from skopt import BayesSearchCV
#from skopt.space import Real, Categorical, Integer


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
              "   Accuracy: " + str(Acc))
    plt.show()
    
    
def plotGrid(data, x_ax, y_ax, title):
    sns.set()
    fig, ax1 = plt.subplots(figsize = (10, 10))
    sns.heatmap(data, annot=True, ax=ax1, cmap="viridis",  fmt=".2%")
    ax1.set_title(title)
    ax1.set_ylabel(x_ax)
    ax1.set_xlabel(y_ax)