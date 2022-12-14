# -*- coding: utf-8 -*-
"""covid19_cases_predict.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UsPSw61HiuIKS9v1bNKGAS7-8c5qGHJz
"""

from google.colab import files
uploaded = files.upload()

import os
import pickle
import datetime
import numpy as np
import pandas as pd 
import seaborn as sns 
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential,Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM,Dropout,Dense

# 1) Data Loading

CSV_PATH_TRAIN = os.path.join(os.getcwd(), 'cases_malaysia_train.csv')
df_train = pd.read_csv(CSV_PATH_TRAIN)

CSV_PATH_TEST = os.path.join(os.getcwd(),'cases_malaysia_test.csv')
df_test = pd.read_csv(CSV_PATH_TEST)

# 2) Data Inspection/Visualization
df_train.head()

df_test.head()

df_train.isna().sum()
# There are null values in this train dataset cluster_import, cluster_religious,cluster_community          
# cluster_highRisk, cluster_education, cluster_detentionCentre, cluster_workplace

df_test.isna().sum()
# There is only one null value in cases_new

df_train.describe().T

df_test.describe().T

df_train.info()

df_test.info()

df_disp = df_train[100:200] # Set display resolution

# Using distplot is not accurate as there's no pattern can be seen
plt.figure()
sns.distplot(df_disp['cases_new'])
plt.show()

plt.figure()
plt.plot(df_disp['cases_new'])
plt.show()

# 3) Data Cleaning
# We will be using only 1 feature from this dataset which cases_new 

# The cases_new column in df_train is in object, and theres a ? and empty rows in the column
# Change the ? and empty rows to nan value 
# Change the data type of this column to float

df_train['cases_new'] = df_train['cases_new'].replace(' ', np.nan)
df_train['cases_new'] = df_train['cases_new'].replace('?', np.nan)

df_train['cases_new']=pd.to_numeric(df_train['cases_new'])

# Since this is a timeseries data to fill the NaN use interpolate method
df_train['cases_new']=df_train['cases_new'].interpolate()

df_train.dtypes
# The cases_new is change to float

# There is 1 NaN value in cases_new column from df_test
df_test.isna().sum()

# Since this is a timeseries data to fill the NaN use interpolate method
df_test['cases_new']=df_test['cases_new'].interpolate()

# 4) Features selection
 # There is only one feature selected which is the cases_new

# 5) Data Preprocessing 

# Train dataset
X = df_train['cases_new'] # use only one feature 

# MinMaxScaler use column???s minimum and maximum value to scale the data series
mms = MinMaxScaler()
X = mms.fit_transform(np.expand_dims(X,axis=-1))

# Set the window size as 30 (30 columns from the train dataset)
win_size = 30
X_train = []
y_train = []

# Will give the X_train & y_train in list, need to be in array
for i in range(win_size,len(X)):
    X_train.append(X[i-win_size:i])
    y_train.append(X[i])

# Change X_train,y_train to array
X_train = np.array(X_train)
y_train = np.array(y_train)

# Test Datasets
#Since test data is too small to be train, do concatenation(combine) with the train data
dataset_concat = pd.concat((df_train['cases_new'],df_test['cases_new']))

# Combine the window size from train dataset with test dataset
length_days = win_size + len(df_test)
tot_input = dataset_concat[-length_days:]

Xtest = mms.fit_transform(np.expand_dims(tot_input,axis=-1))
# win_size = 30

X_test = []
y_test = []

# Will give the X_test & y_test in list, need to be in array
for i in range(win_size,len(Xtest)):
    X_test.append(Xtest[i-win_size:i])
    y_test.append(Xtest[i])

# Change X_train,y_train to array
X_test = np.array(X_test)
y_test = np.array(y_test)

# 5) Model Dvelopment

# Use LSTM as it  capable of learning long-term dependencies, especially in sequence prediction problems
# Use relu as the activation fucntion as it is one proven as a good activation function

input_shape = np.shape(X_train)[1:]

model = Sequential()
model.add(Input(shape=(input_shape))) 
model.add(LSTM(64,return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(64,return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1,activation='relu'))

model.summary()

plot_model(model,show_shapes=True,show_layer_names=True)

model.compile(optimizer='adam',loss='mse',
              metrics=['mean_absolute_percentage_error'])

#Tensorboard
log_dirs = os.path.join("logs_fits",datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 

tensorboard_callback = TensorBoard(log_dir=log_dirs, histogram_freq=1)

#Model training
hist = model.fit(X_train,y_train,
                 epochs=280,
                 callbacks=[tensorboard_callback],
                 validation_data=(X_test,y_test))

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs_fits

#Model Evaluation

predicted_cases = model.predict(X_test)

plt.figure()
plt.plot(y_test,color='red', label='Actual')
plt.plot(predicted_cases,color='blue', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()

actual_cases = mms.inverse_transform(y_test)
predicted_cases_in = mms.inverse_transform(predicted_cases)

plt.figure()
plt.plot(actual_cases,color='red', label='Actual')
plt.plot(predicted_cases_in,color='blue', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()

print(model.evaluate(X_test,y_test))

# Calculate the mean absolute percentage error
mean_absolute_percentage_error(actual_cases,predicted_cases_in)

# Model Saving

# MMS
OHE_SAVE_PATH = os.path.join(os.getcwd(),'mms.pkl')
with open(OHE_SAVE_PATH,'wb') as file:
  pickle.dump(mms,file)

# Model
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
model.save(MODEL_SAVE_PATH)