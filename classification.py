# classification
from google.colab import drive
drive.mount('/content/drive')

import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

dataset = pd.read_csv('/content/drive/MyDrive/weatherAUS.csv')
print(dataset.shape)
categorical_features = dataset.select_dtypes(include=['object']).columns
numerical_features = dataset.select_dtypes(include=['int64','float64']).columns
print(categorical_features)
print(numerical_features)

dataset.dtypes

categorical_percent_missing = dataset[categorical_features].isnull().mean()
numerical_percent_missing = dataset[numerical_features].isnull().mean()
print(categorical_percent_missing)
print(numerical_percent_missing)

print(dataset['Sunshine'].isnull())

dataset.shape

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
dataset[numerical_features] = imputer.fit_transform(dataset[numerical_features])

numerical_percent_missing = dataset[numerical_features].isnull().mean()
print(numerical_percent_missing)

print(dataset.shape)

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(dataset.iloc[:,[1,7,9,10,21,22]])
dataset.iloc[:,[1,7,9,10,21,22]] = imputer.transform(dataset.iloc[:,[1,7,9,10,21,22]])
dataset

categorical_percent_missing = dataset[categorical_features].isnull().mean()
print(categorical_percent_missing)

dataset.shape

import matplotlib.pyplot as plt
import seaborn as sns
for col in numerical_features:
    sns.boxplot(x=dataset[col])
    plt.show()

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1
    low  = q1 - 1.5*iqr
    high = q3 + 1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > low) & (df_in[col_name] < high)]
    return df_out

dataset.shape

temp=dataset.copy()

for i in numerical_features:
  temp=remove_outlier(temp,i)
  print(temp.shape)

"""Data preparation

"""

from sklearn.preprocessing import LabelEncoder
dataset2=dataset.copy()
labelencoder_X = LabelEncoder()
for i in [0,1,7,9,10,21,22]:
  dataset.iloc[:, i] = labelencoder_X.fit_transform(dataset.iloc[:, i])
dataset

X =  dataset.iloc[:,0:-1]
Y = dataset.iloc[:,-1]

Y.head

X

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [1,7,9,10,21])], remainder='passthrough')
X = ct.fit_transform(X)
X

"""Standardization

"""

X = pd.DataFrame(X)
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

accuracy = []
preci=[]
recall = []
f1 = []
algo_liste=[]
def algodetails(name,rep):
  algo_liste.append(name)
  accuracy.append(rep['accuracy'])
  preci.append(rep['macro avg']['precision'])
  recall.append(rep['macro avg']['recall'])
  f1.append(rep['macro avg']['f1-score'])

"""Logistic Regression"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred,output_dict=True)
algodetails("Logistic Regression",report)

"""Seaborn Heatmap"""

import seaborn as sb
sb.heatmap(cm, annot=True,cmap="YlGnBu",fmt ='d')

"""Naive BAYES

"""

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

# Train the model using the training sets
model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix

# Predict the response for test dataset
y_pred3 = model.predict(X_test)

# Generate the confusion matrix
con = confusion_matrix(y_test, y_pred3)

report = classification_report(y_test, y_pred3,output_dict=True)
algodetails("Naive Bayes",report)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(con, annot = True, fmt = 'd')

accuracy_score(y_test, y_pred3)

print(classification_report(y_test, y_pred3))

"""Decision Tree"""

from sklearn.tree import DecisionTreeClassifier
dest = DecisionTreeClassifier(max_depth=50)
dest.fit(X_train, y_train)
y_pred = dest.predict(X_test)

print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred,output_dict=True)
algodetails("Decision Tree",report)

"""Neural Networks"""

from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)

print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred,output_dict=True)
algodetails("Neural Networks",report)

"""Support Vector Machines"""

from sklearn.svm import LinearSVC
svma = LinearSVC()
svma.fit(X_train, y_train)
y_pred = svma.predict(X_test)

print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred,output_dict=True)
algodetails("SVM",report)

cm = confusion_matrix(y_test, y_pred)
cm

"""Hyperparameter Tuning"""

from sklearn.model_selection import GridSearchCV

accuracyH = []
preciH=[]
recallH = []
f1H = []
algo_listeH=[]
def algodetailsH(name,rep):
  algo_listeH.append(name)
  accuracyH.append(rep['accuracy'])
  preciH.append(rep['macro avg']['precision'])
  recallH.append(rep['macro avg']['recall'])
  f1H.append(rep['macro avg']['f1-score'])

"""Logistic Regression"""

from sklearn.model_selection import RepeatedStratifiedKFold
HLR = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(C=c_values)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=HLR, param_grid=grid, n_jobs=-1, cv=5, scoring='accuracy',verbose=2)
grid_result = grid_search.fit(X_train, y_train)

bestHLR = LogisticRegression(**grid_search.best_params_,random_state=42)
bestHLR.fit(X_train,y_train)
y_pred = bestHLR.predict(X_test)

print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred,output_dict=True)
algodetailsH("Log reg H",report)

"""Naive Bayes

"""

params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
nbH = GaussianNB()
gs_NB = GridSearchCV(estimator=nbH, 
                     param_grid=params_NB, 
                     cv=3,
                     verbose=2, 
                     scoring='accuracy')
gs_NB.fit(X_train, y_train)
bestNBH = GaussianNB(**gs_NB.best_params_)
bestNBH.fit(X_train, y_train)
y_pred = bestNBH.predict(X_test)

print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred,output_dict=True)
algodetailsH("Naive Bayes H",report)

"""Decision Trees"""

DTCH = DecisionTreeClassifier(random_state=123)
#
# Create grid parameters for hyperparameter tuning
#
params =  {
    'min_samples_leaf': [1, 2, 3],
    'max_depth': [5,10,50]
}
#
# Create gridsearch instance
#
grid = GridSearchCV(estimator=DTCH,
                    param_grid=params,
                    cv=3,
                    n_jobs=1,
                    verbose=2)
#
# Fit the model
#
grid.fit(X_train, y_train)
bestDTCH = DecisionTreeClassifier(**grid.best_params_,random_state=123)
bestDTCH.fit(X_train, y_train)
y_pred = bestDTCH.predict(X_test)

print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred,output_dict=True)
algodetailsH("Decision Tree H",report)

"""Neural Networks"""

mlp = MLPClassifier()
parameter_space = {
     
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']
}
clf = GridSearchCV(mlp, parameter_space,cv=2,verbose=2)
clf.fit(X_train, y_train)
bestmlp = MLPClassifier(**clf.best_params_,random_state=123)
bestmlp.fit(X_train, y_train)
y_pred = bestmlp.predict(X_test)

print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred,output_dict=True)
algodetailsH("Neural Networks H",report)

"""SVM"""

param_grid = {'C': [0.1, 1], 
              'loss': ['hinge', 'squared_hinge']}
grid = GridSearchCV(LinearSVC(), param_grid, refit = True,cv=2, verbose = 3)
grid.fit(X_train, y_train)
bestsvm  = LinearSVC(**grid.best_params_)
bestsvm.fit(X_train, y_train)
y_pred = bestsvm.predict(X_test)

print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred,output_dict=True)
algodetailsH("SVM H",report)

score={"algo_list":algo_liste,"Accuracy":accuracy,"precision":preci,"recall":recall,"f1_score":f1}
scoreH={"algo_list":algo_listeH,"Accuracy":accuracyH,"precision":preciH,"recall":recallH,"f1_score":f1H}

score = pd.DataFrame(score)
scoreH = pd.DataFrame(scoreH)

"""Algorithms before Hyper Parameter tuning"""

score

"""Algorithms after Hyper Parameter tuning"""

scoreH