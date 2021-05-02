import csv
from sklearn import tree
from sklearn.utils import resample
import matplotlib.pyplot as plt

db = []
with open("../resource/forestfires.csv", 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         #print(row)

for k in range(5):  
  X = []
  Y = []
  bootstrapSample = resample(db, n_samples=len(db), replace=True)
  
  for instance in bootstrapSample:
      X.append(instance[4:-1])
      if float(instance[-1]) > 0:
          Y.append(1)
      else:
          Y.append(0)
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
  clf = clf.fit(X, Y)
  
  num_correct = 0
  for x,y in zip(X,Y):
      class_predicted = clf.predict([x])[0]
      num_correct += int(class_predicted == y)
  accuracy = num_correct / len(db)

  print(accuracy);
  tree.plot_tree(clf, feature_names=['FFMC','DMC','DC','ISI','temp','RH','wind','rain'], class_names=['no_fire','fire'], filled=True, rounded=True)
  plt.show()
  text_representation = tree.export_text(clf, feature_names=['FFMC','DMC','DC','ISI','temp','RH','wind','rain'])
  print(text_representation)

X = []
Y = []
for instance in db:
  temp = []
  temp.append(instance[5])
  temp.append(instance[8])
  temp.append(instance[10])
  X.append(temp)
  if float(instance[-1]) > 0:
      Y.append(1)
  else:
      Y.append(0)

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

num_correct = 0
for x,y in zip(X,Y):
  class_predicted = clf.predict([x])[0]
  num_correct += int(class_predicted == y)
accuracy = num_correct / len(db)

print(accuracy);
#tree.plot_tree(clf, feature_names=['DMC','temp','wind'], class_names=['no_fire','fire'], filled=True, rounded=True)
#plt.show()




