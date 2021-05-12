#Library
import csv
import math
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

#Variables
X = []
y = []
features = ('X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain')
month_dict = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
              'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,}
day_dict = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}
#modifiable
fs_rounds = None
final_dim = None
function = None

#Retrieve user input
fs_rounds = int(input("Enter # of feature selection rounds: "))
final_dim = int(input("Enter # of desired features: "))
print("Choose a feature selection function:\n1. correlation\n2. mutual information")
choice = int(input("Enter 1 or 2: "))
if choice == 1:
    function = f_regression
if choice == 2:
    function = mutual_info_regression

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

#Feature Selection
print("\nPerforming feature selection...", end=' ')
fs_totals = [0,0,0,0,0,0,0,0,0,0,0,0]
for fs_round in range(fs_rounds):
    fs = SelectKBest(score_func=function, k=final_dim)
    fs.fit(X_std, y)
    for i,score in enumerate(fs.scores_):
        fs_totals[i] += score
fs_avgscores = [i/fs_rounds for i in fs_totals]
print("Done")

#Determine columns of selected features
keep_col = sorted(range(len(fs_avgscores)), key=lambda i: fs_avgscores[i], reverse=True)[:final_dim]
print("\nUse feature columns: " + str(keep_col))




