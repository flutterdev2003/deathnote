import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import Normalizer 
from datetime import datetime

dataset = pd.read_csv('avocado.csv')
X =  dataset.drop([dataset.columns[0],'AveragePrice'], axis = 1) #all columns except index column and Average price
Y = dataset.AveragePrice #any average price

X = X.iloc[:,:]
Y=Y.iloc[:]

print(X)

print(Y)

X.isnull().sum()

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])
X.iloc[:, -1] = labelencoder_X.fit_transform(X.iloc[:, -1])
X.iloc[:, 10] = labelencoder_X.fit_transform(X.iloc[:, 10])
X.iloc[:, 9] = labelencoder_X.fit_transform(X.iloc[:, 9])

print(X)

print(X)

dataset.region.unique()

X.region.dtype

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=0), [9,10])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X.shape)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

X_train

X_test

y_train

y_test

sc = StandardScaler()
X_train[:,7:] = sc.fit_transform(X_train[:,7:])
X_test[:,7:] = sc.transform(X_test[:,7:])
X_train[:,:6] = sc.fit_transform(X_train[:,:6])
X_test[:,:6] = sc.transform(X_test[:,:6])

X_train

X_test

X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)

X_train.describe()

X_test.describe()

X_train.to_csv("aa.csv")
X_train.to_csv("bb.csv")

import numpy as np
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(X_train, y_train)

clf.predict([X_test.iloc[13,:]])

y_test.iloc[13]