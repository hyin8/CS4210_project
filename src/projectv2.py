#Libray
import csv
import random
from sklearn import tree
from sklearn import svm
from sklearn.utils import resample
import matplotlib.pyplot as plt

#Variables
db = []
test_set = []
train_set = []
#modifiable
fs_rounds = 20
test_proportion = 0.25
feature_number = 3
depth = 5

#Load data
with open("../resource/forestfires.csv", 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
          temp = []
          for k,value in enumerate(row):
              if k > 3: #ignore the first four features
                  temp.append(float(value))
          db.append(temp)
          
#Split data into test and train
train_set = db.copy()
split_test_total = round(len(train_set)*test_proportion)
for instance in range(split_test_total):
  random_index = random.randint(0,len(train_set)-1)
  test_set.append(train_set[random_index])
  train_set.remove(train_set[random_index])
    
#Feature Selection 
feature_frequencies = [0,0,0,0,0,0,0,0]
features = ['FFMC','DMC','DC','ISI','temp','RH','wind','rain']
for k in range(fs_rounds):
  X_training = []
  Y_training = []
  bootstrapSample = resample(train_set, n_samples=len(train_set), replace=True)
  
  for instance in bootstrapSample:
      X_training.append(instance[:-1])
      Y_training.append(instance[-1])
      
  dtr = tree.DecisionTreeRegressor(max_depth = 2)
  dtr = dtr.fit(X_training, Y_training)
  
  text_representation = tree.export_text(dtr, feature_names=features)
  for i,feature in enumerate(features):
      feature_frequencies[i] += text_representation.count(feature)/2
      
print("Feature frequencies: " + str(feature_frequencies))

#Determine columns of selected features
top_features = sorted(range(len(feature_frequencies)), key=lambda i: feature_frequencies[i], reverse=True)[:feature_number]
keep_col = [i+3 for i in top_features]
print("Use feature columns: " + str(keep_col))

#Update training set according to selected features
X_training = []
Y_training = []
for instance in train_set:
  temp = []
  for k,value in enumerate(instance):
      if k in keep_col:
          temp.append(float(value))
  X_training.append(temp)
  Y_training.append(instance[-1])

#Populate testing set according to selected features
X_testing = []
Y_testing = []
for instance in test_set:
  temp = []
  for k,value in enumerate(instance):
      if k in keep_col:
          temp.append(float(value))
  X_testing.append(temp)
  Y_testing.append(instance[-1])
  
#Determine best parameters for svr
print("\nSVR testing:")
c = [1, 5, 10]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
lowestMAD = 20
parameters = ()
for c_value in c: #iterates over c
  for degree_value in degree: #iterates over degree
      for kernel_type in kernel: #iterates kernel
          svr = svm.SVR(C=c_value, degree=degree_value, kernel=kernel_type, gamma='scale')
          svr.fit(X_training,Y_training)
          
          total_error = 0
          for x,y in zip(X_testing,Y_testing):
              predicted_value = svr.predict([x])[0]
              total_error += abs(y - predicted_value)
          mad = total_error/len(test_set)
          if mad < lowestMAD:
              lowestMAD = mad
              parameters = (c_value, degree_value, kernel_type)
              print("MAD: %f, Parameters: C=%d, degree=%d, kernel=%s" % (mad, c_value, degree_value, kernel_type))

#Build final models
dtr = tree.DecisionTreeRegressor(max_depth=5)
dtr = dtr.fit(X_training, Y_training)
svr = svm.SVR(C=parameters[0], degree=parameters[1], kernel=parameters[2], gamma='scale')
svr.fit(X_training,Y_training)

#Test models
Y_series = [] #used to save actual values and respective predictions
dtr_error = 0
svr_error = 0
for x,y in zip(X_testing,Y_testing):
  dtr_prediction = dtr.predict([x])[0]
  svr_prediction = svr.predict([x])[0]
  Y_series.append([y,dtr_prediction,svr_prediction])
  dtr_error += abs(y - dtr_prediction)
  svr_error += abs(y - svr_prediction)
dtr_mad = dtr_error/len(test_set)
svr_mad = svr_error/len(test_set)
print("\nDTR w/ depth " + str(depth) + ":\t\tMAD = " + str(dtr_mad))
print("SVR w/ best parameters:\tMAD = " + str(svr_mad))

#Sort actual values and predictions by increasing order of actual values
Y_series.sort(key=lambda x:x[0])
truth_series = [i for i,j,k in Y_series] #actual values
dtr_series = [j for i,j,k in Y_series] #dtr predictions
svr_series = [k for i,j,k in Y_series] #svr predictions

#Plot actual values and dtr predictions
plt.figure(figsize=(12,12))
plt.title(label="DTR and Actual")
plt.plot(truth_series, 'bo')
plt.plot(dtr_series, 'r+')

#Plot actual values and svr predictions
plt.figure(figsize=(12,12))
plt.title(label="SVR and Actual")
plt.plot(truth_series, 'bo')
plt.plot(svr_series, 'r+')



