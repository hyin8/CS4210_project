#Libray
import csv
import math
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

#Variables
X = []
y = []
Y_series = [] #used to save actual values and respective predictions
dtr_MADscores = []
svr_MADscores = []
dtr_RMSEscores = []
svr_RMSEscores = []
features = ('X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain')
month_dict = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
              'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,}
day_dict = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}
#modifiable
fs_rounds = 20
kf_rounds = 10
final_dim = 4
depth = 5

#Load data
with open("../resource/forestfires.csv", 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
          temp = [row[0],row[1],month_dict[row[2]],day_dict[row[3]],row[4],row[5],
                  row[6],row[8],row[9],row[10],row[11]]
          X.append([float(i) for i in temp])
          #natural log transform function
          y.append(math.log(float(row[-1])+1))

#Feature Selection
fs_totals = [0,0,0,0,0,0,0,0,0,0,0,0]
for fs_round in range(fs_rounds):
    fs = SelectKBest(score_func=f_regression, k=final_dim)
    fs.fit(X, y)
    for i,score in enumerate(fs.scores_):
        fs_totals[i] += score
fs_avgscores = [i/fs_rounds for i in fs_totals]

#Determine columns of selected features
keep_col = sorted(range(len(fs_avgscores)), key=lambda i: fs_avgscores[i], reverse=True)[:final_dim]
print("Use feature columns: " + str(keep_col))

#Update X according to selected features
X_fs = []
for instance in X:
    temp = []
    for k,value in enumerate(instance):
        if k in keep_col:
            temp.append(value)
    X_fs.append(temp)

#Split data into test and train
for kf_round in range(kf_rounds):
    print("Round " + str(kf_round+1))
    kf = KFold(n_splits=10, shuffle=True)
    kf.split(X_fs,y)
    for train_index, test_index in kf.split(X):
        X_train = []
        y_train = []
        for index in train_index:
            X_train.append(X[index])
            y_train.append(y[index])
        X_test = []
        y_test = []
        for index in test_index:
            X_test.append(X[index])
            y_test.append(y[index])
    
        X_train_std = preprocessing.scale(X_train)
        X_test_std = preprocessing.scale(X_test)
        
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
                    svr.fit(X_train_std,y_train)
                  
                    total_error = 0
                    for instance,target in zip(X_test_std,y_test):
                        predicted_value = math.exp(svr.predict([instance])[0])-1
                        target_transform = math.exp(target)-1
                        total_error += abs(target_transform - predicted_value)
                    mad = total_error/len(X_test_std)
                    if mad < lowestMAD:
                        lowestMAD = mad
                        parameters = (c_value, degree_value, kernel_type)
        
        #Build models
        dtr = tree.DecisionTreeRegressor(max_depth=depth)
        dtr = dtr.fit(X_train_std,y_train)
        svr = svm.SVR(C=parameters[0], degree=parameters[1], kernel=parameters[2], gamma='scale')
        svr.fit(X_train_std,y_train)
        
        #Test models
        dtr_totalError = 0
        svr_totalError = 0
        dtr_totalSE = 0
        svr_totalSE = 0
        for instance,target in zip(X_test_std,y_test):
            #reversing natural log transformation
            dtr_prediction = math.exp(dtr.predict([instance])[0])-1
            svr_prediction = math.exp(svr.predict([instance])[0])-1
            target_transform = math.exp(target)-1
            #populating series for later plotting
            Y_series.append([target_transform,dtr_prediction,svr_prediction])
            #totaling errors
            dtr_error = target_transform - dtr_prediction
            svr_error = target_transform - svr_prediction
            dtr_totalError += abs(dtr_error)
            svr_totalError += abs(svr_error)
            dtr_totalSE += dtr_error*dtr_error
            svr_totalSE += svr_error*svr_error
        #saving MAD and RMSE scores
        dtr_MADscores.append(dtr_totalError/len(X_test_std))
        svr_MADscores.append(svr_totalError/len(X_test_std))
        dtr_RMSEscores.append(math.sqrt(dtr_totalSE/len(X_test_std)))
        svr_RMSEscores.append(math.sqrt(svr_totalSE/len(X_test_std)))

#Calculte average scores
dtr_avgMAD = sum(dtr_MADscores)/len(dtr_MADscores)
svr_avgMAD = sum(svr_MADscores)/len(svr_MADscores)
dtr_avgRMSE = sum(dtr_RMSEscores)/len(dtr_RMSEscores)
svr_avgRMSE = sum(svr_RMSEscores)/len(svr_RMSEscores)

#Print average peformace of models
print("\nDTR w/ depth " + str(depth) +":")
print("average MAD = " + str(dtr_avgMAD))
print("average RMSE = " + str(dtr_avgRMSE))

print("\nSVR w/ best parameters:")
print("average MAD = " + str(svr_avgMAD))
print("average RMSE = " + str(svr_avgRMSE))

#Sort actual values and predictions by increasing order of actual values
Y_series.sort(key=lambda x:x[0])
truth_series = [i for i,j,k in Y_series] #actual values
dtr_series = [j for i,j,k in Y_series] #dtr predictions
svr_series = [k for i,j,k in Y_series] #svr predictions

#Plot actual values and dtr predictions
plt.figure(figsize=(12,12))
plt.title(label="DTR")
plt.xlabel('Ordered test set')
plt.ylabel('Burned area (in hectares)')
plt.ylim([0,20])
plt.plot(truth_series, 'bo', label='real values')
plt.plot(dtr_series, 'r+', label='predictions')

#Plot actual values and svr predictions
plt.figure(figsize=(12,12))
plt.title(label="SVR")
plt.xlabel("Ordered test set")
plt.ylabel("Burned area (in hectares)")
plt.ylim([0,20])
plt.plot(truth_series, 'bo', label='real values')
plt.plot(svr_series, 'r+', label='predictions')



