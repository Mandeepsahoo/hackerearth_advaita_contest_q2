import numpy as np
import pandas as pd
from random import randint
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score

train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('Test.csv')
#plot a random sample
items = ['T-shirt or top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

a = randint(0,10000)
data = train.ix[a]
label = data[0]
pixels = data[1:]

pixels = np.array(pixels, dtype='uint8')
pixels = pixels.reshape((28,28))
plt.title('Label is {}'.format(items[label]))
plt.imshow(pixels, cmap = 'gray')
plt.show()




y_train = train.label.values.astype('int32')
train = train.drop('label', axis=1)
X_train = train.values.astype('float32')
X_test = test.values.astype('float32')



#normalizing
input_dim = X_train.shape[1]
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()



#modelling

nn = Sequential()
nn.add(Dense(64, input_dim = input_dim, activation = 'relu'))
nn.add(Dropout(0.2))
nn.add(Dense(128, activation = 'relu'))
nn.add(Dropout(0.3))
nn.add(Dense(128, activation = 'relu'))
nn.add(Dropout(0.3))
nn.add(Dense(128, activation = 'relu'))
nn.add(Dense(10, activation = 'softmax'))

sgd = SGD(lr = 0.08, momentum = 0.9, decay = 1e-5, nesterov=True)
nn.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
nn.fit(X_train, y_train, epochs = 150,batch_size = 128, shuffle=True, verbose = 2)


preds = nn.predict(X_test)
preds = np.argmax(preds, axis = 1)
result = pd.DataFrame(preds)
result.to_csv('predic.csv', index=False, header=False)
