#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 


from sklearn.svm import SVC

#for c in [10, 100, 1000, 10000]:
for c in [10000]:
	print 'C: %s' % c
	clf = SVC(kernel='rbf', C=c)
	start_time = time()
	clf.fit(features_train, labels_train)
	print 'train time %.3f' % (time() - start_time)

	start_time = time()
	pred = clf.predict(features_test)
	print 'pred time %.3f' % (time() - start_time)

	from sklearn.metrics import accuracy_score
	print accuracy_score(labels_test, pred)

print sum(pred)
#print pred[10], pred[26], pred[50]


#########################################################
### your code goes here ###

#########################################################


