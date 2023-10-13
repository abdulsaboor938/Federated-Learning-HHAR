# code file for applying RBF SVM on the data

# importing all the libraries and depenedencies

import os

# !pip install -qqq pandas
os.system('pip install -qqq pandas')
# !pip install -qqq numpy
os.system('pip install -qqq numpy')
# !pip install -qqq scikit-learn
os.system('pip install -qqq scikit-learn')

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import math


# disabling warnings
import warnings
warnings.filterwarnings('ignore')

# Data preprocessing

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

# loading the data

def load_data(path):
    df_one=pd.read_csv(path)
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

# list the paths of all files in the extasensory_data folder
import os
path="extrasensory_dataset"
files=os.listdir(path)
files.sort()

# make complete path of all files
for i in range(len(files)):
    files[i]=path+"/"+files[i]

# training function

def train(dataset):
    
    # loading data
    X,y=load_data(dataset)

    # making a single column for all the activities
    y['activity']=y.idxmax(axis=1)

    # convering the categorical data into numerical data
    y=y['activity'].replace({'label:FIX_running':0,'label:FIX_walking':1,'label:SITTING':2,'label:SLEEPING':3,'label:OR_standing':4})

    # Split data into training and testing sets
    X_train, X_test,\
        y_train, y_test = train_test_split(X,
                                            y,
                                            test_size=0.2)
    
    # Standardize features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create an SVM classifier with an RBF
    # kernel and set values of C and gamma
    model = SVC(kernel='rbf', C=200, gamma=100)

    # Fit the model to the training data
    model.fit(X_train_scaled, y_train)

    # Calculate the accuracy of the model on the test data
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return { "dataset": dataset, "accuracy": accuracy, "model": model, "support_vectors": model.support_vectors_, "dual_coef": model.dual_coef_}

def main():
    # run model for all the datasets and store the results in a text file

    results = []

    for i in range(len(files)):
        try:
            result = train(files[i])
            results.append(result)
            print(f"{i}: {result['dataset']} - {result['accuracy']}")
        except:
            print(f"Error in {files[i]}")
            pass

    with open('svm_results.txt', 'w') as f:
        f.write(str(results))

if __name__ == "__main__":
    main()