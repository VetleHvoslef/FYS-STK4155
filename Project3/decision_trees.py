import numpy as np
import PreprosessingAstroids as pp
import Models as m


#Data type
AsteroidClass = 1
Hazardious = 0
Feature = "all" # 'all', 'select', 'drop'

# Only relevant for Hazaroud asteroid classification
# How to handel imbapanced data: "RUS", "OS", "US"
S = "RUS"

#Model
DecisionTree = 0
RandomForest = 0
GBoost = 1


#AsteroidClass (Classifier)
if AsteroidClass:
    print("AsteroidClass, Multiclass classificationon")
    DataType = "Asteroid Class"
    X_train_scaled, X_test_scaled, y_train, y_test, CL_label = pp.get_classi_data()

#Hazardious (Classifier)
if Hazardious:
    print("Hazardious Astroids , Binary Classification")
    DataType = "Hazardious asteroides"
    X_train_scaled, X_test_scaled, y_train, y_test = pp.get_haz_data(Feature,S)
  
#............................................................. 



####################################
#      Decicion tree
####################################
if DecisionTree:
    print("\n Run Decision Tree")
    MaxDepth = np.linspace(1,9,9)
    
    for i in range(len(MaxDepth)):

        dtc, tmp1, tmp2 = m.run_DecisionTree(X_train_scaled, y_train, 
                                             X_test_scaled, y_test, 
                                             int(MaxDepth[i]), DataType)
        





####################################
#      Random Forest 
####################################
if RandomForest:
    print("\n Run Random Forest")
    N_Estimators = np.linspace(50,200,4)
    LearningRate = np.logspace(-5, -1, 5)
    # make loop over maxdepth insteda of learnrate (also uncoment in for-loop)
    #maxdepth = np.linspace(3,9,7)
    MaxDepth = 4

    rfc_Train_accuracy=np.zeros((len(N_Estimators),len(LearningRate)))
    rfc_Test_accuracy=np.zeros((len(N_Estimators),len(LearningRate)))  
    
    for i in range(len(N_Estimators)):
        for j in range(len(LearningRate)): # Comment out this if loop over MaxDepth
        #for k in range(len(maxdepth)):
            #j=0
            #MaxDepth = int(maxdepth[k])
            rfc, tmp1, tmp2 = m.run_RandomForest(X_train_scaled, y_train, 
                                                 X_test_scaled, y_test, 
                                                 int(N_Estimators[i]), 
                                                 LearningRate[j], MaxDepth, DataType)
            rfc_Train_accuracy[i][j] = tmp1
            rfc_Test_accuracy[i][j] =  tmp2

            
    #Make gridplpot of accuracy for different learnrate and number of estimators        
    T1 = DataType + ", Random Forest, " + ", Eta: " + str(LearningRate[j]) +". Estim: " + str(N_Estimators[i])    
    m.plotGrid(rfc_Test_accuracy, "log(learn_rate)", "N estimators", T1+" Test accuracy" )
    m.plotGrid(rfc_Train_accuracy, "log(learn_rate)", "N estimators", T1+" Train accuracy")
    
    
    
#.............................................................   


####################################
#      GBoost 
####################################
if GBoost:    
    print("Run GBoost")
    n_est = 100
    eta = 0.001
    depth = 3
    #No loop since the GBoost takes long time to run. 
    gbc, Train_acc_gbc, Test_acc_gbc = m.run_GradientBoost(X_train_scaled, y_train, 
                                                            X_test_scaled, y_test, 
                                                            n_est, eta, depth, DataType)
    

