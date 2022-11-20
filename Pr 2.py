#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')


# In[2]:


#import dataset and split into train and test data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


plt.matshow(x_train[1])


# In[4]:


plt.imshow(-x_train[0], cmap="gray")


# In[5]:


x_train = x_train / 255
x_test = x_test / 255


# In[6]:


model = keras.Sequential([
keras.layers.Flatten(input_shape=(28, 28)),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dense(10, activation="softmax")
])

model.summary()


# In[7]:


model.compile(optimizer="sgd",
loss="sparse_categorical_crossentropy",
metrics=['accuracy'])


# In[8]:


history=model.fit(x_train,
y_train,validation_data=(x_test,y_test),epochs=10)


# In[9]:


test_loss,test_acc=model.evaluate(x_test,y_test)
# print("Loss=%.3f" %test_loss)
# print("Accuracy=%.3f" %test_acc)


# In[10]:


n=random.randint(0,9999)
plt.imshow(x_test[n])
plt.show()


# In[11]:


x_train


# In[12]:


x_test


# In[13]:


predicted_value=model.predict(x_test)
plt.imshow(x_test[n])
plt.show()

# print(predicted_value[n])


# In[14]:


# history.history()
history.history.keys()
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[15]:


# history.history()
history.history.keys()
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:




