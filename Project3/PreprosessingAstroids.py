import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from  sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import missingno as msno



def SplitScale(X,Y, random_state=0):
    X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size=0.2, 
                                                         random_state=0)
    if random_state is None:
            X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
    


def get_classi_data(random_state=0, return_X_y=False, no_plotting=False):
    inputs = getData(no_plotting=no_plotting)
    
    #Encode output class
    y = inputs.loc[:, 'class']
    le = LabelEncoder()
    le.fit(y)
    CL_label = list(le.classes_) # returns array with labels for each class
    print("Asteroids classes:", CL_label)
    #CL = le.transform(CL_label) #returns array with corresponding number for each class
    Y = le.transform(y)

    if not(no_plotting):
        # Plot to see class diststribution
        plt.plot(Y,'*')
        plt.title("Asteroid class distribution")
        plt.ylabel("Class number")
        plt.xlabel("Astroids")
        plt.show()
    
    # Feature matrix
    inputs = inputs.drop(['class'], axis='columns')
    X = inputs.loc[:, inputs.columns]

    if return_X_y:
        return X, y

    print("Split and Scale the data")
    X_train_scaled, X_test_scaled, y_train, y_test  = SplitScale(X, Y, random_state=0)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, CL_label



def get_haz_data(fet, S, random_state=0, return_X_y=False, no_plotting=False):
    inputs = getData(no_plotting=no_plotting)
    
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
    
    if not(no_plotting):
        plt.figure()
        plt.plot(inputs[inputs['pha'] == 1]['class'],'*')
        plt.title("Class distribution of hazardious astroids in dataset")
        plt.show()
    
    #### Under-sampling #####
    if S == "US":
        print("Under-Sampling")
        class_0_under = class_0.sample(class_count_1)
        test_under = pd.concat([class_0_under, class_1], axis=0)
        print("\n Distribution after Resampling:")
        print(test_under['pha'].value_counts())
        # plot the count after under-sampeling
        #test_under['pha'].value_counts().plot(kind='bar', title='count (target)')      
        inputs = test_under  
        
    elif S =="OS":
        print("Over-Sampling")
        class_1_over = class_1.sample(class_count_0, replace=True)
        test_over = pd.concat([class_1_over, class_0], axis=0)
        print("\n Distribution after Resampling:")
        print(test_over['pha'].value_counts())
        inputs = test_over
        
        
    elif S == "RUS":
        print("Random Under-Sampling")
        print('Original dataset shape:', np.shape(inputs))
        from imblearn.under_sampling import RandomUnderSampler
        # fit predictor and target variable
        rus = RandomUnderSampler(random_state=0, replacement=True)
        x = inputs.drop(['pha'], axis='columns')
        x_rus, y_rus = rus.fit_resample(x, inputs[:]['pha'])
        inputs = x_rus.assign(pha = y_rus)
        print('Resample dataset shape', np.shape(inputs))

    elif S == "unbalanced":
        pass

    
    # Crate output for hazzardios asteroids
    y = inputs[:]['pha']
    inputs = inputs.drop(['pha'], axis='columns')
    
    #All features
    if fet == 'all':
        # Feature matrix
        X = inputs.loc[:, inputs.columns]
     
    #Selected fetures
    elif fet == 'select':
        
        # Analysis to choos most important features
        f,pf = f_classif(inputs, y)
        
        #Select the features with best score
        #saves features and scores in rising order
        d = {'features':inputs.columns , 'classif':f}
        df = pd.DataFrame(data=d)
        #replace Inf with O
        df.replace([np.inf, -np.inf], 0, inplace=True)
        #sort values by their score
        a = df.sort_values(by=['classif'],ascending=False)
        # choos the 8 best features
        b = a[0:8]['features']
        print("Most important features (analyses of variance): ", list(b))
        
        #Keep the best features for further analysis
        X = inputs[:][b] 

        
           
    # exclude moid(distance to earth)  and H (max magnitude)
    # the best features to identify hazardous asteroids.
    elif fet == 'drop':
        inputs = inputs.drop(['moid'], axis='columns')
        X = inputs.drop(['H'], axis='columns')

    if return_X_y:
        return X, y

    
    #Print out feauter-labels
    print("Features: ", X.columns)
    print(" ")
    print("Split and Scale the data")
    X_train_scaled, X_test_scaled, y_train, y_test  = SplitScale(X, y, random_state=0)
    
    if not(no_plotting):
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
        plt.title("Class distribution of astroids in dataset")
        plt.show()
    return X_train_scaled, X_test_scaled, y_train, y_test



def getData(no_plotting=False):
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
    if not(no_plotting):
        inputs_numerical = inputs.drop("MBA")
        correlation_matrix = inputs_numerical.corr().round(1)
        plt.figure(figsize=(15,8))
        sns.heatmap(data=correlation_matrix, annot=True)
        plt.title("Correlation matrix for Astroid data")
        plt.show()
    #important ones from correlation matrix: per, moid, a, ad, q  
    return inputs



