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
   
print('Load files')
inputs = pd.read_csv('dataset.csv', low_memory=False)

#Print out feauter-labels
#print(inputs.columns)


print("Formatting data")
# Removes duplicate rows from the DataFrame
item0 = inputs.shape[0]
inputs = inputs.drop_duplicates() 
item1 = inputs.shape[0]
print("There are ",item0-item1, " duplicates found in the dataset")

#NaN-analyses
N_nan = inputs.isnull().sum(axis = 0)
msno.matrix(inputs) 


corr_inputs = inputs.drop("class", axis=1) # All the values in that column was MBA or so, not numeric
correlation_matrix = corr_inputs.corr().round(1)
# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
plt.figure(figsize=(15,8))
sns.heatmap(data=correlation_matrix, annot=True)
plt.title("Correlation matrix for Astroid data")
plt.show()

# remove some data thats not interesting for further alaysis.
inputs = inputs.drop(['id', 'spkid', 'full_name', 'pdes', 'name', 'prefix', 
                      'orbit_id', 'diameter','albedo', 'diameter_sigma',
                      'equinox', 'per_y', 'moid_ld', 'tp_cal',
                       'epoch_mjd', 'epoch_cal'], axis='columns')
#Most left out because they are not properties that can be measured but parameters single out eache one of them 
# Some features are left out becaus its equal to another feature, but with another unit.
#   ie: per and per_y, first with unit days and  the other in years.  
# Equinox is a string with month of year and has been left out for convinience (enough other data)
# Some are left out because of lack of measurements, to many NaN (as diameter and albedo)


#Remove rows containing NaN in dataset 
# remove all rows that contain one NaN, lead to compleet 
#loss of data from two astroid classes (HYA and IEO)
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



#print(inputs[0:10]['pha'],inputs[0:10]['moid'])

# Create correlation matrix
corr_inputs = inputs.drop("class", axis=1) # All the values in that column was MBA or so, not numeric
correlation_matrix = corr_inputs.corr().round(1)
# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
plt.figure(figsize=(15,8))
sns.heatmap(data=correlation_matrix, annot=True)
plt.title("Correlation matrix for Astroid data")
plt.show()
#important ones from correlation matrix: per, moid, a, ad, q


#Encode output class
y = inputs.loc[:, 'class']
le = LabelEncoder()
le.fit(y)
CL_label = list(le.classes_) # returns array with labels for each class
CL = le.transform(CL_label) #returns array with corresponding number for each class
Y = le.transform(y) # Er dette det samme som le.fit(y)??
print("Astroid classes: ", CL_label)
print("Encoded output: ", CL)
plt.plot(Y,'*')


# Crate output for hazzardios astroids
#y_haz = inputs.loc[:,'pha'] # do not need to encode; {0,1}
#X_haz = inputs[[:],['moid'][:],['diameter']]

inputs = inputs.drop(['class'], axis='columns')
X = inputs.loc[:, inputs.columns]



# Analysis to choos most important features
#c,pc = chi2(X, y) # need only positiv inputs in X
f,pf = f_classif(X, y)

pca = PCA(n_components=0.97)
X2D = pca.fit_transform(X)

#Select the features with best score
d = {'features':inputs.columns , 'classif':f}
df = pd.DataFrame(data=d)
df.replace([np.inf, -np.inf], 0, inplace=True)
a = df.sort_values(by=['classif'],ascending=False)
b = a[0:8]['features']
feat = inputs[b]
print("Most important features: ", list(b), "(ANOVA-test)")




"""

N = inputs.loc[inputs['pha'] == 0]
N = inputs.drop(['pha'], axis='columns')
H = inputs.loc[inputs['pha'] == 1]
H = inputs.drop(['pha'], axis='columns')
fig, axes = plt.subplots(15,2,figsize=(10,20))
ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:,i], bins =50)
    ax[i].hist(malignant[:,i], bins = bins, alpha = 0.5)
    ax[i].hist(benign[:,i], bins = bins, alpha = 0.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["
              Non-Hazzardous", "Hazzardous"], loc ="best")
fig.tight_layout()
plt.show()

"""

#plt.plot(inputs[:]['pha'],'*')


print("Split and Scale the data")
X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size=0.3, 
                                                     random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("Run GBoost")
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
print("1")

# Train the classifier on the training data
gbc.fit(X_train_scaled, y_train)
print("2")

# Make predictions on the test data
y_pred = gbc.predict(X_test_scaled)
print("3")

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
