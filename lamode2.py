# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0:1].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 390, init = 'uniform', activation = 'relu', input_dim = 784))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 390, init = 'uniform', activation = 'relu'))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 390, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, y, batch_size = 10, nb_epoch = 100)

test_set = pd.read_csv('Test.csv')
test_set = sc.transform(test_set)

y_pred = classifier.predict(test_set)
y_pred = np.argmax(y_pred, axis =1)

