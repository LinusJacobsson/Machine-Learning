#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model as skl_lm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import scipy.stats

# We first isolate training data and test data.

np.random.seed(1)
data = pd.read_csv('train.csv')
y_features = "Lead"
LR_model = LogisticRegression(max_iter = 10000) # Create an instance of the model

x_features_all = ["Number words female", "Total words", "Number of words lead",
          "Difference in words lead and co-lead", "Number of male actors",
          "Year","Number of female actors","Number words male","Gross",
          "Mean Age Male","Mean Age Female","Age Lead","Age Co-Lead"]

x_features_omitted = ["Total words", "Gross"]

x_features_all = ["Number words female", "Total words", "Number of words lead",
          "Difference in words lead and co-lead", "Number of male actors",
          "Year","Number of female actors","Number words male","Gross",
          "Mean Age Male","Mean Age Female","Age Lead","Age Co-Lead"]

x_features = [feature for feature in x_features_all if feature not in x_features_omitted]

y_data = data[y_features]
x_data = data[x_features]

#splits data into train and test data
#x_tv, x_test, y_tv, y_test = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.01)
x_tv = x_data # We don't set aside testdata, we are provided with that later on
y_tv = y_data

y_data.replace("Female", 1, inplace = True)
y_data.replace("Male", 0, inplace = True)

# We use k-fold cross validation for determining hyperparameters
k_fold = 20
# USe built in sklearn.modelselection Kfold
k_folder = sklearn.model_selection.KFold(n_splits=k_fold, shuffle=True) # this permutes and splits data

training_errors = np.zeros((k_fold, 1))
validation_errors = np.zeros((k_fold, 1))
feature_count = len(x_features)

for i, (train_index, validation_index) in enumerate(k_folder.split(x_data)):

    x_train = x_tv.iloc[train_index]
    y_train = y_tv.iloc[train_index]

    x_eval = x_tv.iloc[validation_index]
    y_eval = y_tv.iloc[validation_index]

    LR_model.fit(x_train, y_train)

    #Training error
    y_train_pred = LR_model.predict(x_train)
    y_train_err = np.mean(y_train_pred != y_train)
    training_errors[i] = y_train_err

    #Evaluation error
    y_eval_pred = LR_model.predict(x_eval)
    y_eval_err = np.mean(y_eval_pred != y_eval)
    validation_errors[i] = y_eval_err


E_training = np.mean(training_errors)
E_kfold = np.mean(validation_errors)

print("Training and evaluation was completed with E_kfold (incorrect/total): " +str(E_kfold) +" and E_training: " + str(E_training))

Error_error = validation_errors.std()/np.sqrt(len(validation_errors))*100
print("Error in k-fold estimate (%): " + str(Error_error))


# In[ ]:




