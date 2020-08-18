import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("train.csv")
x = dataset.iloc[:, [1, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15]].values
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 16].values
a = dataset["drop_loc"]
b = dataset["total_amount"]
plt.ylim(-0.2, 1000)
plt.scatter(a,b)
dataset.dtypes
s = "vendor_id"
dataset[s] = pd.to_numeric(dataset[s], errors = 'coerce')
s = "driver_tip"
dataset[s] = pd.to_numeric(dataset[s], errors = 'coerce')
s = "mta_tax"
dataset[s] = pd.to_numeric(dataset[s], errors = 'coerce')
s = "toll_amount"
dataset[s] = pd.to_numeric(dataset[s], errors = 'coerce')
s = "extra_charges"
dataset[s] = pd.to_numeric(dataset[s], errors = 'coerce')
s = "improvement_charge"
dataset[s] = pd.to_numeric(dataset[s], errors = 'coerce')
s = "total_amount"
dataset[s] = pd.to_numeric(dataset[s], errors = 'coerce')

z = "pickup_time"
dataset[z] = pd.to_datetime(dataset[z])

z = "drop_time"
dataset[z] = pd.to_datetime(dataset[z])

# Missing data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "mean", axis=0)
imputer = imputer.fit(y)
y = imputer.transform(y)

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 13] = labelencoder_x.fit_transform(x[:, 13])

onehotencoder = OneHotEncoder(categorical_features = [13])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# split dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the Regression Model to the dataset
