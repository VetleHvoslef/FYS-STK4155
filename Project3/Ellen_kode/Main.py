import numpy as np
import PreprosessingAstroids as pp
import Models as m
from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_cumulative_gain
from sklearn.metrics import RocCurveDisplay, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Main code
np.random.seed(0)

#Data type
AsteroidClass = 0
Hazardious = 1
Feature = "select" # 'all', 'select', 'one'

#Model
GBoost = 0
RandomForest = 1
# DecisionTree


#AsteroidClass (Classifier)
if AsteroidClass:
    print("AsteroidClass, Multiclass classificationon")
    DataType = "Asteroid Class"
    X_train_scaled, X_test_scaled, y_train, y_test, CL_label = pp.get_classi_data()

#Hazardious (Classifier)
if Hazardious:
    print("Hazardious Astroids , Binary Classification")
    DataType = "Hazardious asteroides"
    # How to handle imbalanced data: "RUS", "OS", "US"
    S = "RUS"

    X_train_scaled, X_test_scaled, y_train, y_test = pp.get_haz_data(Feature,S)
  
#............................................................. 



####################################
#      Random Forest 
####################################
if RandomForest:
    print("Run Random Forest")
    N_Estimators = np.linspace(200,200,1)
    LearningRate = np.logspace(-4, -4, 1)
    MaxDepth = 3   
    
    rfc_Train_accuracy=np.zeros((len(N_Estimators),len(LearningRate)))
    rfc_Test_accuracy=np.zeros((len(N_Estimators),len(LearningRate))) 
    
    for i in range(len(N_Estimators)):
        for j in range(len(LearningRate)):
            rfc, tmp1, tmp2 = m.run_RandomForest(X_train_scaled, y_train, 
                                                 X_test_scaled, y_test, 
                                                 int(N_Estimators[i]), 
                                                 LearningRate[j], MaxDepth)
            rfc_Train_accuracy[i][j] = tmp1
            rfc_Test_accuracy[i][j] =  tmp2
            y_prob = rfc.predict_proba(X_test_scaled)
            
            #Make plots
            T1 = DataType + ", Random Forest, " + ", Eta: " + str(LearningRate[j]) +". Estim: " + str(N_Estimators[i])
            m.plotTree(rfc, tmp2, N_Estimators[i], LearningRate[j], MaxDepth)
            m.plotROC(y_test, y_prob,T1+" ROC")
            
            if AsteroidClass:
                m.plotMultiConfusion(y_test, rfc.predict(X_test_scaled),T1+" Confusion Matrix")

            if Hazardious:
                m.plotConfusion(y_test, rfc.predict(X_test_scaled),T1+" Confusion Matrix")    
                m.plotGain(y_test, y_prob,T1+" Cumulative Gain")
        
    m.plotGrid(rfc_Test_accuracy, "log(learn_rate)", "N estimators", T1+" Test accuracy" )
    m.plotGrid(rfc_Train_accuracy, "log(learn_rate)", "N estimators", T1+" Train accuracy")
    
    
    
#.............................................................   


####################################
#      GBoost 
####################################
# Accuracy: 99.99%, n_est = 100, eta = 10**-3, depth = 3
if GBoost:    
    print("Run GBoost")
    n_est = 100
    eta = 1E-3
    depth = 3
    #HisGradientBoostClassifier Large dataset
    gbc, Train_acc_gbc, Test_acc_gbc = m.run_GradientBoost(X_train_scaled, y_train, 
                                                            X_test_scaled, y_test, 
                                                            n_est, eta, depth)
    
    #Make plots
    T1 = DataType + ", GradientBoost, " + ", Eta: " + str(eta) +". Estim: " + str(n_est) + ", Depth: " + str(depth)
    m.plotTree(gbc, tmp2, N_Estimators[i], LearningRate[j], MaxDepth)
    m.plotROC(y_test, y_prob,T1+"ROC")
    
    if AsteroidClass:
        m.plotMultiConfusion(y_test, gbc.predict(X_test_scaled),T1+"Confusion Matrix")

    if Hazardious:
        m.plotConfusion(y_test, gbc.predict(X_test_scaled),T1+"Confusion Matrix")    
        m.plotGain(y_test, y_prob,T1+"Cumulative Gain")

