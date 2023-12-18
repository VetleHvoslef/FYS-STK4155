import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

from scikitplot.metrics import plot_roc
from sklearn.metrics import multilabel_confusion_matrix

####################
#   Desicion tree
####################
def run_DecisionTree(X_train_scaled, y_train, X_test_scaled, y_test, depth, DataType):

    dtc = tree.DecisionTreeClassifier(max_depth = depth, random_state = 0)
    
    # Train the classifier on the training data
    dtc.fit(X_train_scaled, y_train)

    # Make predictions on the test data
    y_pred_train = dtc.predict(X_train_scaled)
    y_pred_test = dtc.predict(X_test_scaled)
    y_prob_train = dtc.predict_proba(X_train_scaled)
    y_prob_test = dtc.predict_proba(X_test_scaled)
    
    # Evaluate the accuracy of the model
    Train_accuracy = accuracy_score(y_train, y_pred_train)
    Test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print("\n Depth: ", depth)
    print(f"Train Accuracy: {Train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {Test_accuracy * 100:.2f}%")
    print(" ")
    
    #Make plots
    T1 = DataType + ", Decision Tree, MaxDepth: " + str(depth)
    
    #plotTree(dtc, Test_accuracy, depth)
    plotROC(y_train, y_prob_train, T1 +" ROC Train")
    plotROC(y_test, y_prob_test, T1 +" ROC Test")
    
    if DataType == "Asteroid Class":
        plt.figure()
        plt.plot(y_pred_test,'*')
        plt.title("Class distribution of hazardious astroids as predicted, Test")
        plt.show()
        
        plotMultiConfusion(y_train, dtc.predict(X_train_scaled),T1+" Confusion Matrix, Train")
        plotMultiConfusion(y_test, dtc.predict(X_test_scaled),T1+" Confusion Matrix, Test")

    if DataType == "Hazardious asteroides":
        plotConfusion(y_train, dtc.predict(X_train_scaled),T1+" Confusion Matrix, Train")  
        plotConfusion(y_test, dtc.predict(X_test_scaled),T1+" Confusion Matrix, Test") 
        #plotGain(y_test, y_prob_test,T1 +" Cumulative Gain")
        
        tmp = precision_recall_fscore_support(y_test, y_pred_test)
        pre, rec, _ = precision_recall_curve(y_test, y_pred_test)
        PreRec(pre,rec, T1+" Precision Recall Curv, " +
                  "  Depth: " + str(depth) + 
                  "  Accuracy: " + str(round(Test_accuracy,3)))
        
        print(f"Precision: {tmp[0][0] * 100 :.2f}%")
        print(f"Recall: {tmp[1][0] * 100 :.2f}%")
        print(f"F1 score: {tmp[2][0] * 100 :.2f}%")
        print(" ")

    return dtc, Train_accuracy, Test_accuracy



####################################
#      Random Forest 
####################################
def run_RandomForest(X_train_scaled, y_train, X_test_scaled, y_test, n_est, eta, depth, DataType):

    rfc = RandomForestClassifier(n_estimators = n_est, max_depth = depth, 
                                 random_state = 0)
    
    # Train the classifier on the training data
    rfc.fit(X_train_scaled, y_train)

    # Make predictions on the test data
    y_pred_train = rfc.predict(X_train_scaled)
    y_pred_test = rfc.predict(X_test_scaled)
    y_prob_train = rfc.predict_proba(X_train_scaled)
    y_prob_test = rfc.predict_proba(X_test_scaled)
    

    # Evaluate the accuracy of the model
    Train_accuracy = accuracy_score(y_train, y_pred_train)
    Test_accuracy = accuracy_score(y_test, y_pred_test)
    print("\n N Estimators: ", int(n_est), " Learning Rate: ", eta)
    print(f"Train Accuracy: {Train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {Test_accuracy * 100:.2f}%")  


    #Make plots
    T1 = DataType + ", Random Forest, " + ", Eta: " + str(eta) +". Estim: " + str(n_est)
    plotForest(rfc, Test_accuracy, n_est, eta, depth)
    plotROC(y_train, y_prob_train, T1 +" ROC Train")
    plotROC(y_test, y_prob_test, T1 +" ROC Test")
    
    if DataType == "Asteroid Class":
        plt.figure()
        plt.plot(y_pred_test,'*')
        plt.title("Class distribution of hazardious astroids as predicted, Test")
        plt.show()
        
        plotMultiConfusion(y_train, rfc.predict(X_train_scaled),T1+" Confusion Matrix, Train")
        plotMultiConfusion(y_test, rfc.predict(X_test_scaled),T1+" Confusion Matrix, Test")

    if DataType == "Hazardious asteroides":
        plotConfusion(y_train, rfc.predict(X_train_scaled),T1+" Confusion Matrix, Train")  
        plotConfusion(y_test, rfc.predict(X_test_scaled),T1+" Confusion Matrix, Test") 

        tmp = precision_recall_fscore_support(y_test, y_pred_test)
        
        pre, rec, _ = precision_recall_curve(y_test, y_pred_test)
        PreRec(pre,rec, T1+" Precision Recall Curv, " +
                  "  Depth: " + str(depth) + 
                  "  Eta: " + str(eta) +
                  "  Esitmator: " + str(n_est) +
                  "  Accuracy: " + str(round(Test_accuracy,3)))
        
        print(f"Precision: {tmp[0][0] * 100 :.2f}%")
        print(f"Recall: {tmp[1][0] * 100 :.2f}%")
        print(f"F1 score: {tmp[2][0] * 100 :.2f}%")

    return rfc, Train_accuracy, Test_accuracy


####################################
#      GBoost 
####################################
def run_GradientBoost(X_train_scaled, y_train, X_test_scaled, y_test, n_est, eta, depth, DataType):
    gbc = GradientBoostingClassifier(n_estimators = n_est, learning_rate = eta, 
                                     max_depth = depth, random_state = 42, verbose=1)
    print("Training GBoost model")
    # Train the classifier on the training data
    gbc.fit(X_train_scaled, y_train)
    
    print("Predict GBoost model")
    # Make predictions on the test data
    y_pred_train = gbc.predict(X_train_scaled)
    y_pred_test = gbc.predict(X_test_scaled)
    y_prob_train = gbc.predict_proba(X_train_scaled)
    y_prob_test = gbc.predict_proba(X_test_scaled)
    
    print("Evaluate GBoost model accuracy")
    # Evaluate the accuracy of the model
    Train_accuracy = accuracy_score(y_train, y_pred_train)
    Test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"GBoost Train Accuracy: {Train_accuracy * 100:.2f}%")
    print(f"GBoost Test Accuracy: {Test_accuracy * 100:.2f}%")
    print("  ")

    #Make plots
    T1 = DataType + ", GradientBoost, " + ", Eta: " + str(eta) +", Estim: " + str(n_est) + ", Depth: " + str(depth)
    plotForest(gbc, Test_accuracy, n_est, eta, depth)
    plotROC(y_train, y_prob_train, T1 +" ROC Train")
    plotROC(y_test, y_prob_test, T1 +" ROC Test")
    
    if DataType == "Asteroid Class":
        plt.figure()
        plt.plot(y_pred_test,'*')
        plt.title("Class distribution of hazardious astroids as predicted, Test")
        plt.show()
        
        plotMultiConfusion(y_train, gbc.predict(X_train_scaled),T1+" Confusion Matrix, Train")
        plotMultiConfusion(y_test, gbc.predict(X_test_scaled),T1+" Confusion Matrix, Test")

    if DataType == "Hazardious asteroides":
        plotConfusion(y_train, gbc.predict(X_train_scaled),T1+" Confusion Matrix, Train")  
        plotConfusion(y_test, gbc.predict(X_test_scaled),T1+" Confusion Matrix, Test") 

        tmp = precision_recall_fscore_support(y_test, y_pred_test)
        
        pre, rec, _ = precision_recall_curve(y_test, y_pred_test)
        PreRec(pre,rec, T1+" Precision Recall Curv, " +
                  "  Depth: " + str(depth) + 
                  "  Eta: " + str(eta) +
                  "  Esitmator: " + str(n_est) +
                  "  Accuracy: " + str(round(Test_accuracy,3)))
        
        print(f"Precision: {tmp[0][0] * 100 :.2f}%")
        print(f"Recall: {tmp[1][0] * 100 :.2f}%")
        print(f"F1 score: {tmp[2][0] * 100 :.2f}%")
    
    return gbc, Train_accuracy, Test_accuracy

####################################
#      MLPClassifier
####################################
def eta_and_lambda_grid(X_train, X_test, y_train, y_test, eta_vals, lmbda_vals, hidden_layers, activation, solver):
    acc_vals = np.zeros((len(eta_vals), len(lmbda_vals)))

    # Do grid search of eta and lambda values
    for i in range(len(eta_vals)):
        for j in range(len(lmbda_vals)):
            network = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver, alpha=lmbda_vals[j], learning_rate_init=eta_vals[i])

            # If it does not converge ignore it, the accuracy value for that will be None or zero 
            with warnings.catch_warnings():
                warnings.filterwarnings(
                        "ignore", category=ConvergenceWarning, module="sklearn"
                        )
                network.fit(X_train, y_train)
            acc_vals[i][j] = network.score(X_test, y_test)
    return acc_vals


def PreRec(pre, rec, T1, filename=None):
    disp = PrecisionRecallDisplay(precision=pre, recall=rec)
    disp.plot()
    plt.title(T1, fontsize=16)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
    plt.savefig(f"plots/{filename}.pdf")


def plotTree(dtc, Acc, depth):
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (25,25), dpi=600)
    tree.plot_tree(dtc)
    plt.title("Desicion Tree, " +
              "   Depth: " + str(depth) + 
              "   Accuracy: " + str(round(Acc,3)), fontsize=16)
    plt.show()


def plotForest(rfc, Acc, n_est, eta, depth):
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (25,25), dpi=600)
    tree.plot_tree(rfc.estimators_[0])
    plt.title("Random Forrest, N_Estimators: " + str(n_est) + 
              "   Learning rate: " + str(eta) + 
              "   Depth: " + str(depth) + 
              "   Accuracy: " + str(round(Acc,3)), fontsize=16)
    plt.show()


def plotGrid(data, x_ax, y_ax, T1, filename=None):
    sns.set()
    sns.set(font_scale=1.4)
    fig, ax1 = plt.subplots(figsize = (10, 10))
    sns.heatmap(data, annot=True, ax=ax1, cmap="viridis",  fmt=".2%")
    ax1.set_title(T1, fontsize=16)
    ax1.set_xlabel(x_ax)
    ax1.set_ylabel(y_ax)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.pdf")
    plt.show()


def plotMultiConfusion(y_test, y_pred,T1):
    # make a confusion matrix for each class
    #no normalization available
    conf_mat = multilabel_confusion_matrix(y_test, y_pred)
    
    #Make a subplot for all classes
    f, axes = plt.subplots(3, 4, figsize=(25, 15))
    axes = axes.ravel()
    for i in range(12):
        disp = ConfusionMatrixDisplay(conf_mat[i][0:2][0:2])
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(f'class {i}')
        #to take away axes cross in matrix
        disp.ax_.tick_params(axis=u'both', which=u'both',length=0)
        disp.ax_.grid(b=None)
        if i<8:
            disp.ax_.set_xlabel('')
        if i%4!=0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()
    
    f.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    f.suptitle(T1, fontsize=26)
    plt.show()


def plotConfusion(y_test, y_pred,T1, filename=None):
    cm = confusion_matrix(y_test, y_pred,normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=['Non-hazardious','Hazardious'])
    disp.plot()
    #to take away axes cross in matrix
    disp.ax_.tick_params(axis=u'both', which=u'both',length=0)
    disp.ax_.grid(visible=None)
    plt.title(T1, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.pdf")
    plt.show()


def plotROC(y_test, y_prob, T1):
    plt.figure()
    plot_roc(y_test, y_prob)
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(T1, fontsize=16)
    plt.legend(fontsize='xx-small')
    plt.show()
