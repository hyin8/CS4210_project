import csv
import os
from sklearn import tree
import matplotlib.pyplot as plt

db = []
X = []
Y = []
with open("../resource/WildFires_DataSet.csv", 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         #print(row)

#print(os.getcwd())
dictLabel = {
  "no_fire":0,
  "fire":1
}

for instance in db:
  X.append(instance[:3])
  Y.append(dictLabel[instance[3]])

clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=5)
clf = clf.fit(X, Y)

tree.plot_tree(clf, feature_names=['NDVI', 'LST', 'BURNED_AREA'], class_names=['fire','no_fire'], filled=True, rounded=True)
plt.show()

print(X)

