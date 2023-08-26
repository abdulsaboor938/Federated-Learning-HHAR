# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:46:33 2023

@author: USMAN ANWER
"""

import flwr as fl
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import math
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from tensorflow.keras.layers import MaxPool1D

M_PI=3.1416
def compute_roll_yaw_pitch(x,y,z):
  #Acceleration around X
  acc_x_accl=[]

  #Acceleration around Y
  acc_y_accl=[]

  #Acceleration arounf Z
  acc_z_accl=[]


  for (x_mean,y_mean,z_mean) in zip(x,y,z):
    acc_x_accl.append(float("{:.2f}".format(x_mean*3.9)))
    acc_y_accl.append(float("{:.2f}".format(y_mean*3.9)))
    acc_z_accl.append(float("{:.2f}".format(z_mean*3.9)))


  acc_pitch=[]
  acc_roll=[]
  acc_yaw=[]

  for (acc_x,acc_y,acc_z) in zip(acc_x_accl,acc_y_accl,acc_z_accl):
    if acc_y==0 and acc_z==0:
      value_pitch=-0.1
    else:
      value_pitch=180 * math.atan (acc_x/math.sqrt(acc_y*acc_y + acc_z*acc_z))/M_PI
    if acc_x ==0 and acc_z==0:
      value_roll=-0.1
      value_yaw=-0.1
    else:
      value_roll = 180 * math.atan (acc_y/math.sqrt(acc_x*acc_x + acc_z*acc_z))/M_PI
      value_yaw = 180 * math.atan (acc_z/math.sqrt(acc_x*acc_x + acc_z*acc_z))/M_PI
    value_pitch=float("{:.2f}".format(value_pitch))
    value_roll=float("{:.2f}".format(value_roll))
    value_yaw=float("{:.2f}".format(value_yaw))
    acc_pitch.append(value_pitch)
    acc_roll.append(value_roll)
    acc_yaw.append(value_yaw)
  return acc_pitch,acc_roll,acc_yaw

#Sliding Window to null values
def fill_null(data):
  for col in data.columns:
    null_indexes=data[data[col].isnull()].index.tolist()
    #print("For ",col)
    for ind in null_indexes:
      #print(" Got null value at ",ind)
      values=data.loc[ind-6:ind-1,col]
      #print(" Last 5 values ",values)
      mean_val=values.mean()
      data.loc[ind,col]=mean_val
  return data


def load_data():
    user_one="C:\\pdata\\extrasensory_dataset\\00EABED2-271D-49D8-B599-1D4A09240601.features_labels.csv.gz"
    df_one=pd.read_csv(user_one)
    #accelerometer
    df_acc=df_one.iloc[:,1:27]
    df_acc=fill_null(df_acc)
    #gyroscope
    df_gyro=df_one.iloc[:,27:53]
    df_gyro=fill_null(df_gyro)
    #magnometer
    df_magnet=df_one.iloc[:,53:84]
    df_magnet=fill_null(df_magnet)
    # watch accelerometer
    #df_watch_acc=df_one.iloc[:,84:130]
    # location
    #df_location=df_one.iloc[:,139:156]
    
    # For accelerometer
    #mean values
    acc_mean_x=df_acc['raw_acc:3d:mean_x']
    acc_mean_y=df_acc['raw_acc:3d:mean_y']
    acc_mean_z=df_acc['raw_acc:3d:mean_z']

    acc_mean_x=acc_mean_x.replace({0:0.001})

    #standard deviations
    #acc_std_x=df_acc['raw_acc:3d:std_x']
    #acc_std_y=df_acc['raw_acc:3d:std_y']
    #acc_std_z=df_acc['raw_acc:3d:std_z']

    (pitch,roll,yaw)=compute_roll_yaw_pitch(acc_mean_x,acc_mean_y,acc_mean_z)
    df_one['acc_pitch']=pitch
    df_one['acc_roll']=roll
    df_one['acc_yaw']=yaw
    
    #for gyroscope
    gyro_mean_x=df_gyro['proc_gyro:3d:mean_x']
    gyro_mean_y=df_gyro['proc_gyro:3d:mean_y']
    gyro_mean_z=df_gyro['proc_gyro:3d:mean_z']

    (pitch,roll,yaw)=compute_roll_yaw_pitch(gyro_mean_x,gyro_mean_y,gyro_mean_z)

    df_one['gyro_pitch']=pitch
    df_one['gyro_roll']=roll
    df_one['gyro_yaw']=yaw

    # For magnetometer
    magno_mean_x=df_magnet['raw_magnet:3d:mean_x']
    magno_mean_y=df_magnet['raw_magnet:3d:mean_y']
    magno_mean_z=df_magnet['raw_magnet:3d:mean_z']

    (pitch,roll,yaw)=compute_roll_yaw_pitch(magno_mean_x,magno_mean_y,magno_mean_z)

    df_one['magno_pitch']=pitch
    df_one['magno_roll']=roll
    df_one['magno_yaw']=yaw
    
    y=df_one[['label:FIX_running','label:FIX_walking','label:SITTING','label:SLEEPING','label:OR_standing']]
    
    # to avoid null values
    y['label:FIX_running']=y['label:FIX_running'].fillna(0)
    y['label:FIX_walking']=y['label:FIX_walking'].fillna(0)
    y['label:SITTING']=y['label:SITTING'].fillna(0)
    y['label:SLEEPING']=y['label:SLEEPING'].fillna(0)
    y['label:OR_standing']=y['label:OR_standing'].fillna(0)
    
    #Check rows where all the recorded activities are zero ~ No activity recorded rows
    list_of_indexs=[]
    for i in range(len(y)):
        run=y.iloc[i,0]
        walk=y.iloc[i,1]
        sit=y.iloc[i,2]
        sleep=y.iloc[i,3]
        stand=y.iloc[i,4]
        activities=[run,walk,sit,sleep,stand]
        count_ones=activities.count(1)
        if walk==0 and run==0 and sit==0 and sleep==0 and stand==0:
            list_of_indexs.append(i)
        #check if more then 1 exists for different activities
        elif count_ones>1:
            list_of_indexs.append(i)
    
    y=y.drop(list_of_indexs)
    X=df_one.iloc[:,-9:]
    X=X.drop(list_of_indexs)
    
    
    return X,y
    
def load_model_1D(X_train):
    # Create sequential model 
    cnn_model = tf.keras.models.Sequential()
    #First CNN layer  with 32 filters, conv window 3, relu activation and same padding
    cnn_model.add(Conv1D(filters=32, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001), input_shape = (X_train.shape[1],1)))
    #Second CNN layer  with 64 filters, conv window 3, relu activation and same padding
    cnn_model.add(Conv1D(filters=64, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    #Third CNN layer with 128 filters, conv window 3, relu activation and same padding
    cnn_model.add(Conv1D(filters=128, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    #Fourth CNN layer with Max pooling
    cnn_model.add(MaxPool1D(pool_size=(3,), strides=2, padding='same'))
    cnn_model.add(Dropout(0.5))
    #Flatten the output
    cnn_model.add(Flatten())
    #Add a dense layer with 256 neurons
    cnn_model.add(Dense(units = 256, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    #Add a dense layer with 512 neurons
    cnn_model.add(Dense(units = 512, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    #Softmax as last layer with five outputs
    cnn_model.add(Dense(units = 5, activation='softmax'))    
    return cnn_model
    
    

# Load model and data (MobileNetV2, CIFAR-10)
#model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
#model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

#load data 
(X,y)=load_data()

#reversing the one-hot encoding
a=np.array(y)
y=np.where(a)[1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

#y_train=np.array(y_train).reshape(y_train.shape[0],y_train.shape[1],1)
#y_test=np.array(y_test).reshape(y_test.shape[0],y_test.shape[1],1)

model=load_model_1D(X_train)

#reshaping the arrays imply CNN
X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# Define Flower client
class CifarClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return model.get_weights()

  def fit(self, parameters, config):
    print(".... Fit Function Called ... ")
    model.set_weights(parameters)
    model.fit(X_train, y_train, epochs=60, batch_size=32)
    return model.get_weights(), len(X_train), {}

  def evaluate(self, parameters, config):
    print(".... Evaluate Function Called ... ")
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(X_train, y_train)
    print("Testing accuracy is ",accuracy)
    return loss, len(X_train), {"accuracy": accuracy}
    

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())