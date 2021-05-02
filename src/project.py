# Library
import csv
import random
from sklearn import tree
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Variables
db = []
test_set = []
train_set = []
classLabel_number = 2
#modifiable
classifer_number = 10
depth = 10
test_proportion = 0.5
remove_col = []

#Load data
with open("../resource/forestfires_clean.csv", 'r') as csvfile:
	reader = csv.reader(csvfile)
	for i, row in enumerate(reader):
		if i > 0: #skipping the header
			temp = []
			for k,value in enumerate(row):
				if k not in remove_col:
					temp.append(value)
			db.append (temp)
			#print(row)

# Dictionary for classifier
dictLabel = {
	"no_fire":0,
	"fire":1
}

#Split data into test and train
train_set = db.copy()
split_test_total = round(len(train_set)*test_proportion)
for instance in range(split_test_total):
	random_index = random.randint(0,len(train_set)-1)
	test_set.append(train_set[random_index])
	train_set.remove(train_set[random_index])

#Initialize voting
class_vote = []
temp = []
for i in range(classLabel_number):
	temp.append(0)
for i in range(split_test_total):
	class_vote.append(temp.copy())

#Base Classifiers section
for k in range(classifer_number): 
	X_training = []
	Y_training = []
	bootstrapSample = resample(train_set, n_samples=len(train_set), replace=True)
	#populate training sets for each classifier
	for instance in bootstrapSample:
		X_training.append(instance[:-1])
		Y_training.append(instance[-1])
	#Train model
	clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=depth)
	clf = clf.fit(X_training, Y_training)
	#Start prediction
	for i,testSample in enumerate(test_set):
		prediction = clf.predict([testSample[:-1]])[0]
		class_vote[i][int(prediction)] += 1

correct = 0
total = 0
for i, testSample in enumerate(test_set):
	total+=1
	true_label = int(testSample[-1])
	guess_label = class_vote[i].index(max(class_vote[i]))
	if true_label == guess_label:
		correct+=1

accuracy = correct/total
print("Accuracy: ",accuracy)

