#Libray
import csv
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import KFold

#Variables
X = []
y = []
Y_series = [] #used to save real values and respective predictions
svr_MADscores = []
svr_RMSEscores = []
features = ('X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain')
month_dict = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
              'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,}
day_dict = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}
#modifiable
keep_col = [0,1,2,8]
kf_rounds = 10

#Retrieve user input
n = int(input("Enter # of desired features: "))
keep_col = list(map(int,input("Enter the feature columns (format: 0 1 2 ...): ").strip().split()))[:n]
kf_rounds = int(input("Enter # of 10 fold cross validation rounds: "))

#Load data
print("\nLoading data...", end=' ')
with open("../resource/forestfires.csv", 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
          temp = [row[0],row[1],month_dict[row[2]],day_dict[row[3]],row[4],row[5],
                  row[6],row[8],row[9],row[10],row[11]]
          X.append([float(i) for i in temp])
          #natural log transform function
          y.append(math.log(float(row[-1])+1))
print("Done")

#Standardize attributes to have a mean of zero and a standard deviation of one
X_std = preprocessing.scale(X)

#Update X according to selected features
X_fs = []
for instance in X_std:
    temp = []
    for k,value in enumerate(instance):
        if k in keep_col:
            temp.append(value)
    X_fs.append(temp)

#Train and test model with rounds of 10 fold cross validtion
print("\nPerforming 10 fold cross validation...")
for kf_round in range(kf_rounds):
    print("Round " + str(kf_round+1))
    #split data into test and train
    kf = KFold(n_splits=10, shuffle=True)
    kf.split(X_fs,y)
    for train_index, test_index in kf.split(X_fs):
        X_train = []
        y_train = []
        for index in train_index:
            X_train.append(X_fs[index])
            y_train.append(y[index])
        X_test = []
        y_test = []
        for index in test_index:
            X_test.append(X_fs[index])
            y_test.append(y[index])
        
        #Determine best parameters for svr
        c = [1, 5, 10]
        degree = [1, 2, 3]
        kernel = ["linear", "poly", "rbf"]
        lowestMAD = 100
        parameters = []
        for c_value in c: #iterates over c
            for degree_value in degree: #iterates over degree
                for kernel_type in kernel: #iterates kernel
                    svr = svm.SVR(C=c_value, degree=degree_value, kernel=kernel_type, gamma='scale')
                    svr.fit(X_train,y_train)
                  
                    total_error = 0
                    for instance,target in zip(X_test,y_test):
                        try:
                            predicted_value = math.exp(svr.predict([instance])[0])-1
                            target_transform = math.exp(target)-1
                            total_error += abs(target_transform - predicted_value)
                        except OverflowError:
                            pass
                    mad = total_error/len(X_test)
                    if mad < lowestMAD:
                        lowestMAD = mad
                        parameters = (c_value, degree_value, kernel_type)
        
        #Build models
        svr = svm.SVR(C=parameters[0], degree=parameters[1], kernel=parameters[2], gamma='scale')
        svr.fit(X_train,y_train)
        
        #Test models
        svr_totalError = 0
        svr_totalSE = 0
        for instance,target in zip(X_test,y_test):
            try:
                #reversing natural log transformation
                svr_prediction = math.exp(svr.predict([instance])[0])-1
                target_transform = math.exp(target)-1
                #populating series for later plotting
                Y_series.append([target_transform,svr_prediction])
                #totaling errors
                svr_error = target_transform - svr_prediction
                svr_totalError += abs(svr_error)
                svr_totalSE += svr_error*svr_error
            except OverflowError:
                pass
        #saving MAD and RMSE scores
        svr_MADscores.append(svr_totalError/len(X_test))
        svr_RMSEscores.append(math.sqrt(svr_totalSE/len(X_test)))
print("Done")

#Calculte average scores
svr_avgMAD = sum(svr_MADscores)/len(svr_MADscores)
svr_avgRMSE = sum(svr_RMSEscores)/len(svr_RMSEscores)

#Print average peformace of models
print("\nSVM w/ best parameters:")
print("average MAD = " + str(svr_avgMAD))
print("average RMSE = " + str(svr_avgRMSE))

#Sort real values and predictions by increasing order of actual values
Y_series.sort(key=lambda x:x[0])
truth_series = [i for i,j in Y_series] #actual values
svr_series = [j for i,j in Y_series] #svr predictions

#Plot real values and svr predictions
plt.figure(figsize=(12,12))
plt.title(label="SVM")
plt.xlabel("Ordered test set")
plt.ylabel("Burned area (in hectares)")
plt.ylim([0,20])
plt.plot(truth_series, 'bo', label='real values')
plt.plot(svr_series, 'r+', label='predictions')
plt.legend()
plt.show()



