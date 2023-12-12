import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from  sklearn.feature_selection import chi2 , f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import missingno as msno 
np.random.seed(0)




def SplitScale(X,Y):
    X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size=0.2, 
                                                         random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
    


def get_classi_data():
    inputs = getData()
    
    #Encode output class
    y = inputs.loc[:, 'class']
    le = LabelEncoder()
    le.fit(y)
    CL_label = list(le.classes_) # returns array with labels for each class
    #CL = le.transform(CL_label) #returns array with corresponding number for each class
    Y = le.transform(y)
    # Plot to see class diststribution
    plt.plot(Y,'*')
    plt.title("Asteroid class distribution")
    plt.ylabel("Class number")
    plt.xlabel("Astroids")
    plt.show()
    
    # Feature matrix
    inputs = inputs.drop(['class'], axis='columns')
    X = inputs.loc[:, inputs.columns]

    print("Split and Scale the data")
    X_train_scaled, X_test_scaled, y_train, y_test  = SplitScale(X,Y)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, CL_label



def get_diam_data(inputs):
    inputs = getData()
    #Feature matrix for regression to decide the diameter
    X = inputs[:]['albedo'],inputs[:]['H']
    y = inputs[:]['diameter']
    X = X.dropna()
    print("Number of data for regression: ", len(X))
    
    print("Split and Scale the data")
    X_train_scaled, X_test_scaled, y_train, y_test  = SplitScale(X,y)
    
    return X_train_scaled, X_test_scaled, y_train, y_test



def get_haz_data(fet, S):
    inputs = getData()
    
    le = LabelEncoder()
    le.fit(inputs.loc[:, 'class'])
    CL_label = list(le.classes_) # returns array with labels for each class
    CL = le.transform(CL_label) #returns array with corresponding number for each class
    inputs['class'] = inputs['class'].replace(CL_label, CL)
        
    class_count_0, class_count_1 = inputs['pha'].value_counts()
    # Separate class
    class_0 = inputs[inputs['pha'] == 0]
    class_1 = inputs[inputs['pha'] == 1]
    # print the shape of the class
    print("\n Distribution before Resampling:")
    print(inputs['pha'].value_counts())
    print("\n Propotion of Potentialy Hazardious Aasteroids in data set: ", 
          round(len(class_1)/len(inputs)*100,2),"%")
    
    
    #### Under-sampling #####
    if S == "undersampling":
        print("Under-Sampling")
        class_0_under = class_0.sample(class_count_1)
        test_under = pd.concat([class_0_under, class_1], axis=0)
        print("\n Distribution after Resampling:")
        print(test_under['pha'].value_counts())
        # plot the count after under-sampeling
        #test_under['pha'].value_counts().plot(kind='bar', title='count (target)')      
        inputs = test_under  
        
    elif S =="oversampling":
        print("Over-Sampling")
        class_1_over = class_1.sample(class_count_0, replace=True)
        test_over = pd.concat([class_1_over, class_0], axis=0)
        print("\n Distribution after Resampling:")
        print(test_over['pha'].value_counts())
        inputs = test_over    
        
    elif S == "RUS":
        print("Random Under-Sampling")
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=0, replacement=True)# fit predictor and target variable
        x_rus, y_rus = rus.fit_resample(inputs, inputs[:]['pha'])

        print('original dataset shape:', np.shape(inputs[:]['pha']))
        print('Resample dataset shape', np.shape(y_rus))

    # Crate output for hazzardios asteroids
    y = inputs[:]['pha']
    inputs = inputs.drop(['pha'], axis='columns')
    
    #All features
    if fet == 'a':
        # Feature matrix
        X = inputs.loc[:, inputs.columns]
     
    #Selected fetures
    elif fet == 's':
           
        # Analysis to choos most important features
        #c,pc = chi2(X, y) # need only positiv inputs in X
        f,pf = f_classif(X, y)
        
        #pca = PCA(n_components=0.97)
        #X2D = pca.fit_transform(X)
        
        #Select the features with best score
        #make a dict of features and scores
        d = {'features':inputs.columns , 'classif':f}
        df = pd.DataFrame(data=d)
        #replace Inf with O
        df.replace([np.inf, -np.inf], 0, inplace=True)
        #sort values by their score
        a = df.sort_values(by=['classif'],ascending=False)
        # choos the 8 best features
        b = a[0:8]['features']
        feat = inputs[b]
        print("Most important features (analyses of variance): ", list(b))
        
        #Keep the best features for further analysis
        X = inputs[:][feat] 
           
    # Just moid
    elif fet == 'o':
        X_1 = np.array(inputs[:]['moid'])
        X = np.reshape(X_1,(len(X_1),1))
        
    print("Split and Scale the data")
    X_train_scaled, X_test_scaled, y_train, y_test  = SplitScale(X,y)
    
    plt.figure()
    plt.plot(y_train,'*')
    plt.title("Distribution of hazardious and Non-hazardious astroids in Train data")
    plt.show()

    plt.figure()
    plt.plot(y_test,'*')
    plt.title("Distribution of hazardious and Non-hazardious astroids in Test data")
    plt.show()
    
    plt.figure()
    plt.plot(inputs[:]['class'],'*')
    plt.title("Distribution of hazardious and Non-hazardious astroids in Test data")
    plt.show()
    
    return X_train_scaled, X_test_scaled, y_train, y_test 



def getData(): 
    print('Load files')
    inputs = pd.read_csv('dataset.csv', low_memory=False)
    
    print("Formatting data")
    # Removes duplicate rows from the DataFrame
    item0 = inputs.shape[0]
    inputs = inputs.drop_duplicates() 
    item1 = inputs.shape[0]
    print("There are ",item0-item1, " duplicates found in the dataset")
    
    #make Data Overview of NAN
    msno.matrix(inputs) 
    
    # remove some data thats not interesting for further alaysis.
    inputs = inputs.drop(['id', 'spkid', 'full_name', 'pdes', 'name', 'prefix', 
                          'orbit_id', 'diameter','albedo', 'diameter_sigma',
                          'equinox', 'per_y', 'moid_ld', 'tp_cal',
                           'epoch_mjd', 'epoch_cal'], axis='columns')

    #Remove rows containing NaN in dataset 
    item0 = inputs.shape[0]
    inputs = inputs.dropna()
    item1 = inputs.shape[0]  
    print("There are ",item0-item1, " rows with NAN removed from the dataset")  
    
    #Replace Categorial values with 0 an 1
    inputs.pha.replace(('Y', 'N'), (1, 0), inplace = True)
    inputs['pha'] = inputs['pha'].fillna(0)
    inputs['pha'] = inputs.pha.astype(int)
    
    inputs.neo.replace(('Y', 'N'), (1, 0), inplace = True)
    inputs['neo'] = inputs['neo'].fillna(0)
    inputs['neo'] = inputs.neo.astype(int)
    
    # Create correlation matrix
    correlation_matrix = inputs.corr().round(1)
    plt.figure(figsize=(15,8))
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.title("Correlation matrix for Astroid data")
    plt.show()
    #important ones from correlation matrix: per, moid, a, ad, q  

    return inputs



 


