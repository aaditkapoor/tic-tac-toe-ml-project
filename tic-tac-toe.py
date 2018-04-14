
# coding: utf-8

# - We are creating a very simple machine learning model.<br>
# - Using dataset: tic-tac-toe.data.txt with user-defined columns.<br>
# - We are treating this problem as a supervised learning problem.<br>

# In[74]:

# This the rough sketch of the processing that happened in my brain while creating the program.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[52]:


# Loading data
data = pd.read_csv("../tic-tac-toe.data.txt", sep = ",")
data_copy = pd.read_csv("../tic-tac-toe.data.txt", sep = ",")

# Setting cols.
data.columns = ["first_row_left", "first_row_middle", "first_row_right", "center_row_left", "center_row_middle", "center_row_right", "bottom_row_left", "bottom_row_middle", "bottom_row_right", "is_win"]
data_copy.columns = ["first_row_left", "first_row_middle", "first_row_right", "center_row_left", "center_row_middle", "center_row_right", "bottom_row_left", "bottom_row_middle", "bottom_row_right", "is_win"]


# In[53]:


# Viewing data
data.head()


# In[54]:


# As we can see the the different move options, we perform label encoding.
mapping_for_moves = {'x':1, "o":0} # For b, we put mean of the data.
mapping_for_wins = {"positive":1, "negative":0} # Positive is win, negative is lose
data.is_win = data.is_win.map(mapping_for_wins)
data_copy.is_win = data_copy.is_win.map(mapping_for_wins)

data = data.drop(columns=["is_win"], axis=1)


# In[55]:


data.head()


# In[56]:


for i in data.columns: # Applying map to all the columns except is_win.
    data[i] = data[i].map(mapping_for_moves)


# In[57]:


data.head() # Viewing data


# In[58]:


# Extracting features and labels
features = data.values
labels = data_copy.is_win.values


# In[63]:


# Filling missing values aka "b"
features = (Imputer().fit_transform(features))


# In[48]:


len(features) 


# In[49]:


len(labels)


# In[65]:


# Changing type to int
features = features.astype(np.int)
labels = labels.astype(np.int)


# In[66]:


features


# In[67]:


labels


# - Preprocessing is done.

# In[68]:


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=3, shuffle=True)


# In[73]:


data.corr()


# - Clearly it is a classification problem, we can use DecisionTree or SVC

# In[84]:


# Trying different classifiers.
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
d_tree_score = clf.score(features_test, labels_test) # Good result!


# In[78]:


clf2 = SVC() # Clearly the data is non linear.
clf2.fit(features_train, labels_train)
clf2.score(features_test, labels_test) # Not good!


# In[85]:


clf3 = KNeighborsClassifier(n_neighbors=1)
clf3.fit(features_train, labels_train)
k_score = clf3.score(features_test, labels_test)


# In[86]:


d_tree_score > k_score


# In[87]:


predictions = clf3.predict(features_test)


# In[89]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, predictions)


# In[90]:


cm


# In[91]:


np.where(labels_test!=predictions)


# In[95]:


d_tree_score


# In[94]:


k_score


# In[97]:


from sklearn.metrics import classification_report
c = classification_report(labels_test, predictions)


# In[98]:


c


# In[115]:


from sklearn.ensemble import RandomForestClassifier
r = RandomForestClassifier(n_estimators=100) # With 100 decision tree
r.fit(features_train, labels_train)
r_forest = r.score(features_test, labels_test)
p = r.predict(features_test)
np.where(labels_test!=features_test) # Only one misclassified


# In[116]:


cm  = confusion_matrix(labels_test, p)


# In[117]:


cm

