
# coding: utf-8

# In[1]:

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

data_bpsk = np.loadtxt('bpsk_5db.txt')
data_qpsk = np.loadtxt('qpsk_5db.txt')
data_16qam = np.loadtxt('qam16_5db.txt')
data_64qam = np.loadtxt('qam64_5db.txt')
print(data_bpsk.shape, data_qpsk.shape, data_16qam.shape, data_64qam.shape)


# In[3]:

data = np.concatenate((data_bpsk, data_qpsk, data_16qam, data_64qam), axis=1)
print(data.shape)


# In[4]:

dataset = data.T
label = np.concatenate((np.ones((1000, 1)), 2 * np.ones((1000, 1)), 
                        3 * np.ones((1000, 1)), 4 * np.ones((1000, 1))), 
                        axis=0)
print(dataset.shape, label.shape)


# In[5]:

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#print(x_train[0].dtype)


# In[6]:

row, col = 32, 32
x_train = x_train.reshape(x_train.shape[0], 3, row, col)
x_test = x_test.reshape(x_test.shape[0], 3, row, col)
print(x_train.shape, x_test.shape)


# In[7]:

input_shape = (3, row, col)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255

print(x_train[0].dtype)


# In[8]:

print(y_train[0:10])
num_classes = 5
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[9]:

print(y_train[0:10])
y_train = np.delete(y_train, 0, 1)
y_test = np.delete(y_test, 0, 1)
print(y_train[0:10])


# In[10]:

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))


# In[11]:

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[12]:

batch_size= 128
epochs = 30
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)


# In[13]:

score = model.evaluate(x_test, y_test)


# In[14]:

print('score: ', score[1])


# In[15]:

print(y_test[0:10])


# In[16]:

get_ipython().magic('matplotlib inline')
plt.figure(figsize=(8, 8))
plt.subplot(1, 4, 1)
#print(x_test[0].shape)
plt.imshow(x_test[1].T)
plt.title('bpsk')
plt.subplot(1, 4, 2)
plt.imshow(x_test[6].T)
plt.title('qpsk')
plt.subplot(1, 4, 3)
plt.imshow(x_test[0].T)
plt.title('16qam')
plt.subplot(1, 4, 4)
plt.imshow(x_test[2].T)
plt.title('64qam')


# In[ ]:



