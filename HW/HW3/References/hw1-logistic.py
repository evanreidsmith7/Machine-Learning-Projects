print('\nImporting Packages........')
# import requirements
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.linear_model import LogisticRegression
print('\t\t\t\t........DONE!')

####################################################
#
# Data Preprocessing
#
####################################################

# Read the Excel file into a DataFrame
print('\nImporting Tabs........')
df_train = pd.read_excel('Data_User_Modeling_Dataset.xls', sheet_name='Training_Data')
df_test = pd.read_excel('Data_User_Modeling_Dataset.xls', sheet_name='Test_Data')
print('\t\t\t\t........DONE!')

# Separating the required columns to input values
print('\nImporting all samples from SCG and STR.......')
X_train = df_train.iloc[:, [1, 2]]
X_test = df_test.iloc[:, [1, 2]]
print('\t\t\t\t........DONE!')

# Replace the string values in the 'UNS' column based on the dictionary
print('\nChanging string class values to numerical ones.......')

# changing the training labels
class_mapping = {'very_low': 0, 'Low': 1, 'Middle': 2, 'High': 3}
df_train[' UNS'] = df_train[' UNS'].replace(class_mapping)
y_train = df_train[' UNS']

# changing the test labels
class_mapping = {'Very Low': 0, 'Low': 1, 'Middle': 2, 'High': 3}
df_test[' UNS'] = df_test[' UNS'].replace(class_mapping)
y_test = df_test[' UNS']
print('\t\t\t\t........DONE!')

# standardize the training and test inputs
print('\nStandardizing the Data.......')
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print('\t\t\t\t........DONE!')

####################################################
#
# Plot the classification outcome using this Method
#
####################################################

def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
	print('\nCreating the Plot Decision figure.......')
	markers = ('s','x','o','D')
	colors = ('red', 'blue', 'lightgreen','orange')
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

	print('\t\t\t\t........DONE!')

####################################################
# End of PLOTTING DECISION CODE
####################################################

####################################################
#
# Model Definition/Training/Testing
#
####################################################
print('\nCreating the Model, Training & Predicting.......')
lr = LogisticRegression(C=10, solver='lbfgs', max_iter=500, multi_class='multinomial', random_state=42)
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)
print('\t\t\t\t........DONE!')

####################################################
#
# Metric Printouts and Text File writes
#
####################################################
print('\nCalculating metrics, generating txt file.......')

#Print out the training and test accuracy values to a text file
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Training Accuracy: %.2f' % lr.score(X_train_std, y_train))
print('Test Accuracy: %.2f' % lr.score(X_test_std, y_test))

#Name of the text file
with open("LR_accuracies.txt", "w") as f:
	f.write('Misclassified samples: %d' % (y_test != y_pred).sum())
	f.write('\n')
	f.write('Training Accuracy: %.2f' % lr.score(X_train_std, y_train))
	f.write('\n')
	f.write('Test Accuracy: %.2f' % lr.score(X_test_std, y_test))
	f.write('\n')
print('\t\t\t\t........DONE!')


####################################################
#
# Decision Plot setup and save image
#
####################################################
print('\nDecision Plot setup and save image.......')

# Combine all training and test data to single object variables
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(y_train.size, y_train.size + y_test.size))

#plot with labels and legend
plt.xlabel('SCG [Standardized]')
plt.ylabel('STR [standardized]')
plt.legend(loc='upper left')

#Save plot ---> LEAP requires this part
plt.savefig("LR_plot.png")

print('\t\t\t\t........DONE!')
print('\n******************PROGRAM is DONE *******************')
