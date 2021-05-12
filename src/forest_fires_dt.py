#Libray
import csv
import math
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import KFold

#Variables
X = []
y = []
Y_series = [] #used to save real values and respective predictions
dtr_MADscores = []
dtr_RMSEscores = []
features = ('X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain')
month_dict = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
              'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,}
day_dict = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}
#modifiable
keep_col = None
kf_rounds = None
depth = None

#Retrieve user input
n = int(input("Enter # of desired features: "))
keep_col = list(map(int,input("Enter the feature columns (format: 0 1 2 ...): ").strip().split()))[:n]
kf_rounds = int(input("Enter # of 10 fold cross validation rounds: "))
depth = int(input("Enter depth of decision tree (3 is reccomended): "))

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

#Update X according to selected features
X_fs = []
for instance in X:#_std:
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
        
        #Build models
        dtr = tree.DecisionTreeRegressor(max_depth=depth)
        dtr = dtr.fit(X_train,y_train)
        
        #Test models
        dtr_totalError = 0
        dtr_totalSE = 0
        for instance,target in zip(X_test,y_test):
            #reversing natural log transformation
            dtr_prediction = math.exp(dtr.predict([instance])[0])-1
            target_transform = math.exp(target)-1
            #populating series for later plotting
            Y_series.append([target_transform,dtr_prediction])
            #totaling errors
            dtr_error = target_transform - dtr_prediction
            dtr_totalError += abs(dtr_error)
            dtr_totalSE += dtr_error*dtr_error
        #saving MAD and RMSE scores
        dtr_MADscores.append(dtr_totalError/len(X_test))
        dtr_RMSEscores.append(math.sqrt(dtr_totalSE/len(X_test)))
print("Done")

#Calculte average scores
dtr_avgMAD = sum(dtr_MADscores)/len(dtr_MADscores)
dtr_avgRMSE = sum(dtr_RMSEscores)/len(dtr_RMSEscores)

#Print average peformace of models
print("\nDT w/ depth " + str(depth) +":")
print("average MAD = " + str(dtr_avgMAD))
print("average RMSE = " + str(dtr_avgRMSE))

#Sort real values and predictions by increasing order of actual values
Y_series.sort(key=lambda x:x[0])
truth_series = [i for i,j in Y_series] #actual values
dtr_series = [j for i,j in Y_series] #dtr predictions

#Plot real values and dtr predictions
plt.figure(figsize=(12,12))
plt.title(label="DT")
plt.xlabel('Ordered test set')
plt.ylabel('Burned area (in hectares)')
plt.ylim([0,20])
plt.plot(truth_series, 'bo', label='real values')
plt.plot(dtr_series, 'r+', label='predictions')
plt.legend()
plt.show()





