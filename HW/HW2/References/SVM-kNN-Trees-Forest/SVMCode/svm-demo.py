# import requirements
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# load and split the IRIS dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# standardize
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


##########################
# Plot the classification outcome using this Method
##########################

def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
	markers = ('s','x','o')
	colors = ('red', 'blue', 'lightgreen')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	# Plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	#Plot all the samples
	X_test,y_test=X[test_idx,:],y[test_idx]
	for idx,cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)

	#Highlight test samples
	if test_idx:
		X_test,y_test =X[test_idx,:],y[test_idx]

	plt.scatter(X_test[:,0],X_test[:,1],facecolors='none', edgecolors='black', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')

##########################
# End of PLOTTING DECISION CODE
##########################
#freq[Hz], Trc1_S11
# The SVM Model
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

# the non linear svm model
svm2 = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm2.fit(X_train_std, y_train)

# ************************************linear******************************************************

#Testing the model data
y_pred = svm.predict(X_test_std)

#Print out the training and test accuracy values to a text file
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Training Accuracy: %.2f' % svm.score(X_train_std, y_train))
print('Test Accuracy: %.2f' % svm.score(X_test_std, y_test))

#Name of the text file
with open("SVM_accuracies.txt", "w") as f:
	f.write('Misclassified samples: %d' % (y_test != y_pred).sum())
	f.write('\n')
	f.write('Training Accuracy: %.2f' % svm.score(X_train_std, y_train))
	f.write('\n')
	f.write('Test Accuracy: %.2f' % svm.score(X_test_std, y_test))
	f.write('\n')

# Combine all training and test data to single object variables
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm, test_idx=range(y_train.size, y_train.size + y_test.size))

#plot with labels and legend
plt.xlabel('Feature 2 [Standardized]')
plt.ylabel('Feature 1 [standardized]')
plt.legend(loc='upper left')

#Save plot ---> LEAP requires this part
plt.savefig("SVM_plot.png")


# ************************************linear******************************************************

# ************************************non-linear******************************************************
#Testing the model data
y_pred2 = svm2.predict(X_test_std)

#Print out the training and test accuracy values to a text file
print('Misclassified samples: %d' % (y_test != y_pred2).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred2))
print('Training Accuracy: %.2f' % svm2.score(X_train_std, y_train))
print('Test Accuracy: %.2f' % svm2.score(X_test_std, y_test))

#Name of the text file
with open("nonlinSVM_accuracies.txt", "a") as f:
	f.write('Misclassified samples: %d' % (y_test != y_pred2).sum())
	f.write('\n')
	f.write('Training Accuracy: %.2f' % svm2.score(X_train_std, y_train))
	f.write('\n')
	f.write('Test Accuracy: %.2f' % svm2.score(X_test_std, y_test))
	f.write('\n')

# Combine all training and test data to single object variables
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm2, test_idx=range(y_train.size, y_train.size + y_test.size))

#plot with labels and legend
plt.xlabel('Feature 2 [Standardized]')
plt.ylabel('Feature 1 [standardized]')
plt.legend(loc='upper left')

#Save plot ---> LEAP requires this part
plt.savefig("nonlinSVM_plot.png")

