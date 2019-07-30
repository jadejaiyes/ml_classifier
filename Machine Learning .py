#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
from sklearn.datasets import load_breast_cancer
# Load dataset
data = load_breast_cancer()
# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
# Look at our data
print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])


# In[2]:


from sklearn.model_selection import train_test_split
# Split our data
train, test, train_labels, test_labels = train_test_split(features,
labels,test_size=0.33,
random_state=42)


# In[6]:


from sklearn.naive_bayes import GaussianNB
#Initialize our classifier
gnb = GaussianNB()
#Train our classifier
model = gnb.fit(train,train_labels)
#Make predictions
pred =gnb.predict(test)
print(pred)


# In[7]:


from sklearn.metrics import accuracy_score
#Evaluate accuracy
print(accuracy_score(test_labels,pred))





