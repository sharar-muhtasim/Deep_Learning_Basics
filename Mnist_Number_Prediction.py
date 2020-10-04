#!/usr/bin/env python
# coding: utf-8

# In[16]:


import tensorflow as tf

mnist=tf.keras.datasets.mnist #28x28 images of written 0-9
(x_train, y_train),(x_test, y_test)= mnist.load_data() #loading the data in these variables

#normlizing the data - scaling the values between 0 and 1 - makes it easier for python
x_train = tf.keras.utils.normalize(x_train, axis=1) 
x_test = tf.keras.utils.normalize(x_test, axis=1)

#building the model architecture
model = tf.keras.models.Sequential() #using a sequential model
model.add(tf.keras.layers.Flatten())#our input layer that we have flattened
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #1st hidden layer - Dense layer with 128 neurons and a relu 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #2nd hidden layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #output layer. 10 nisi so that possible output neuron. Softmax for probability distribution

#parameters to train the model 
model.compile (optimizer="adam", #adam is the go-to. Another example is gradient descent.
              loss="sparse_categorical_crossentropy",#eta default go to
              metrics=["accuracy"]) #what metrics you wanna follow
#the nn is always trying to minimize loss 

#training the model
model.fit(x_train, y_train, epochs=3) #3 bar dataset er upor diye ghurbe



# In[17]:


#checking if we have overfit or underfit
#calculating validation loss and validation accuracing
val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss, val_acc)
#loss would be slightly lower and accuracy would be slightly higher for best results


# In[20]:


#saving a model
model.save("epic_num_reader.model")
#loading a model
new_model = tf.keras.models.load_model("epic_num_reader.model")


# In[39]:


#making a prediction
predictions = new_model.predict([x_test])

#predict always always takes a list, so [] dewa lagbe


# In[48]:


#print(predictions) #eta ekta pixel array er list dibe. Eta ke num banabo

import numpy as np
print(np.argmax(predictions[2042])) #finds the argument and gives the maximum value. Highest probability kisher.


# In[47]:


#checking if the value is true using matplotlib
import matplotlib.pyplot as plt
plt.imshow(x_test[2042]) #displays data as image on a 2D raster
plt.show()

