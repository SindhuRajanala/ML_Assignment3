#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plot


# In[93]:


daf=pd.read_csv("train.csv")


# In[94]:


daf.head()


# In[95]:


le = preprocessing.LabelEncoder()
daf['Sex'] = le.fit_transform(daf.Sex.values)
daf['Survived'].corr(daf['Sex'])


# In[96]:


daf.corr().style.background_gradient(cmap="Greens")


# In[97]:


matrix = daf.corr()
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plot.show()


# In[98]:


train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

train_raw['train'] = 1
test_raw['train'] = 0
daf = train_raw.append(test_raw, sort=False)
features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'
daf = daf[features + [target] + ['train']]
daf['Sex'] = daf['Sex'].replace(["female", "male"], [0, 1])
daf['Embarked'] = daf['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
train = daf.query('train == 1')
test = daf.query('train == 0')


# In[99]:


train.dropna(axis=0, inplace=True)
labels = train[target].values


train.drop(['train', target, 'Pclass'], axis=1, inplace=True)
test.drop(['train', target, 'Pclass'], axis=1, inplace=True)


# In[100]:


from sklearn.model_selection import train_test_split, cross_validate

X_train, X_val, Y_train, Y_val = train_test_split(train, labels, test_size=0.2, random_state=1)


# In[101]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# In[102]:


classifier = GaussianNB()
classifier.fit(X_train, Y_train)


# In[103]:


y_pred = classifier.predict(X_val)
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))

from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# In[80]:


glass=pd.read_csv("glass.csv")


# In[81]:


glass.head()


# In[83]:


glass.corr().style.background_gradient(cmap="Reds")


# In[89]:


matrix = glass.corr()
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plot.show()


# In[90]:


features = ['Rl', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
target = 'Type'


X_train, X_val, Y_train, Y_val = train_test_split(glass[::-1], glass['Type'],test_size=0.2, random_state=1)

classifier = GaussianNB()

classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_val)

print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# In[86]:


from sklearn.svm import SVC, LinearSVC

classifier = LinearSVC()

classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_val)

print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# In[ ]:




