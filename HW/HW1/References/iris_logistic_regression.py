###############################################
#
#   Header Part goes here
#
###############################################

# importing all required packages
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

#Data split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, 
	test_size = 0.3, random_state = 1, stratify = y)

#Scaling training data
sc = StandardScaler()
sc.fit(X_train)
sc.fit(X_test)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Create an instance of Logistic Regression Classifier and fit the data.
lr = LogisticRegression(C=100, random_state=1)

#This is training the model
lr.fit(X_train_std, y_train)

#Testing the model data
y_pred = lr.predict(X_test_std)

#Print out the training and test accuracy values to a text file
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Training Accuracy: %.2f' % lr.score(X_train_std, y_train))
print('Test Accuracy: %.2f' % lr.score(X_test_std, y_test))

#Name of the text file
with open("LogicalRegression_accuracy.txt", "w") as f:
    f.write('Misclassified samples: %d' % (y_test != y_pred).sum())
    f.write('\n')
    f.write('Training Accuracy: %.2f' % lr.score(X_train_std, y_train))
    f.write('\n')
    f.write('Test Accuracy: %.2f' % lr.score(X_test_std, y_test))
    f.write('\n')


##########################
#PLOTTING CODE - pg 56
##########################

#Plot the classification outcome using this Method
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
        plt.scatter(X_test[:,0],X_test[:,1],c='',alpha=1.0,linewidths=1,marker='o',s=55,label='test set')

##########################
# End of PLOTTING DECISION CODE 
##########################

# Combine all training and test data to single object variables
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr)

#plot with labels and legend
plt.xlabel('Feature 2 [Standardized]')
plt.ylabel('Feature 1 [standardized]')
plt.legend()

#Save plot ---> LEAP requires this part
plt.savefig("LogicalRegression_plot.png")
 
