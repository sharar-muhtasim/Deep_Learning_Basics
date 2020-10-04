#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option("display.max_columns", 20)
print("Pandas initialized")


# In[18]:


import numpy as np

# Make numpy values easier to read.
#np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
print("Tensorflow initialized")

# In[4]:


scores = pd.read_csv(r"F:\Machine Learning Shit My Own\StudentsPerformance.csv")
scores=scores[["math score", "reading score","writing score"]]
#scores.head()


# In[5]:


scores_features = scores.copy()
scores_labels = scores_features.pop("writing score")
#scores_labels.head()


# In[6]:


scores_features = np.array(scores_features)
#print(scores_features)


# In[7]:


normalize = preprocessing.Normalization()
normalize.adapt(scores_features)
print("Normalized")


# In[8]:


scores_model = tf.keras.Sequential([
  normalize,
  layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
  layers.Dense(1, activation=tf.nn.relu)
])

scores_model.compile (optimizer="RMSprop", #adam is the go-to. Another example is gradient descent.
              loss=tf.losses.MeanSquaredError(),#eta default go to
              metrics=["mean_absolute_error"]) #what metrics you wanna follow

scores_model.fit(scores_features, scores_labels, epochs=200)


# In[34]:


y=[]
math_score = 30
for reading_score in range(40,60):
    predictions = scores_model.predict([[math_score,reading_score]])
    writing_score = int(predictions)
    y.append(writing_score)
    #print("Reading score: ", reading_score, "\t Writing score: ", int(writing_score))
    

print(y)


# In[38]:


import matplotlib.pyplot as plt

x=[]
for reading_score in range(40,60):
    x.append(reading_score)

print(x)


plt.plot(x, y)

plt.xlabel('Reading Score')
plt.ylabel('Writing Score')

plt.title('Graph')

plt.show()

