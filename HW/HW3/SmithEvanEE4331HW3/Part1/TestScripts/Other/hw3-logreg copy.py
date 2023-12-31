#************************************************************************************
# Evan Smith
# ML – HW#3
# Filename: hw3-logreg.py
# Due: , 2023
#
# Objective:
#*************************************************************************************
# Import libraries
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
#from mlxtend.plotting import plot_decision_regions

##########################################################################################################################
# plot func
###########################################################################################################################
import matplotlib.pyplot as plt
test_idx=None
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
##########################################################################################################################
# Data Preprocessing
###########################################################################################################################

# Set the path of each of the folders we want to excract from
Corridor_rm155_71_loc0000_path = r'Datasets/Measurements_Upload/Corridor_rm155_7.1/Loc_0000'
Lab139_71_loc0000_path =         r'Datasets/Measurements_Upload/Lab139_7.1/Loc_0000'
Main_Lobby71_loc0000_path =      r'Datasets/Measurements_Upload/Main_Lobby_7.1/Loc_0000'
Sport_Hall_71_loc0000_path =     r'Datasets/Measurements_Upload/Sport_Hall_7.1/Loc_0000'

# Create a dataframe to store the data
combined_data = pd.DataFrame()

# Loop through each file in the corridor folder and read the data into a dataframe
for filename in os.listdir(Corridor_rm155_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Corridor_rm155_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data1 = pd.concat([combined_data, data], ignore_index=True)

# add the labels to the dataframe
combined_data1['label'] = 'Corridor'
combined_data1.to_csv("Datasets/combined_data_Corridor_rm155_71_loc0000.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Lab139_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Lab139_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data2 = pd.concat([combined_data, data], ignore_index=True)

combined_data2['label'] = 'Lab139'
combined_data2.to_csv("Datasets/combined_data_Lab139_71_loc0000.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Main_Lobby71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Main_Lobby71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data3 = pd.concat([combined_data, data], ignore_index=True)

combined_data3['label'] = 'Main_Lobby'
combined_data3.to_csv("Datasets/combined_data_Main_Lobby71_loc0000.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Sport_Hall_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Sport_Hall_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data4 = pd.concat([combined_data, data], ignore_index=True)

combined_data4['label'] = 'Sport_Hall'
combined_data4.to_csv("Datasets/combined_data_Sport_Hall_71_loc0000.csv", index=False)

# List of CSV files to combine
csv_files = [
    'Datasets/combined_data_Corridor_rm155_71_loc0000.csv',
    'Datasets/combined_data_Lab139_71_loc0000.csv',
    'Datasets/combined_data_Main_Lobby71_loc0000.csv',
    'Datasets/combined_data_Sport_Hall_71_loc0000.csv'
]

# combine all combined data into a single dataframe
data_frames = [combined_data1, combined_data2, combined_data3, combined_data4]

combined_data = pd.concat(data_frames, ignore_index=True)

combined_data.to_csv("Datasets/combined_data_all.csv", index=False)

# drop the 5th collumn
combined_data = combined_data.drop(columns=[5])

# Assuming you have loaded your data into the combined_data DataFrame
# If not, you should load your data here

# Handling missing values by dropping rows with missing values
combined_data.dropna(inplace=True)

# Splitting into X (features) and y (labels)
X = combined_data.drop(columns=['label'])  # Assuming 'label' is the column containing your labels

# Convert empty strings to NaN
X[X == ''] = np.nan

# Convert all values to float
X = X.astype(float)

#y = combined_data.drop(columns=[0,1,2,3,4])
y = combined_data['label']

# encode y
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(X)
print(X.shape)
print(y_encoded)
print(y_encoded.shape)
'''


##########################################################################################################################
# PIPELINES
###########################################################################################################################
pipe1 = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', LDA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression())
])
param_grid1 = {
    'reduce_dim__n_components': [1, 2, 3],  # Number of components for LDA
    'classifier__penalty': [None, 'l2'],  # Regularization penalty (L1 or L2)
    'classifier__C': [0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
    'classifier__solver': ['sag','lbfgs', 'saga'],  # Solver algorithms
    'classifier__max_iter': [100, 250, 500]  # Maximum number of iterations
}

##########################################################################################################################
# GridSearchCV


gs = GridSearchCV(estimator=pipe_svc,
 ... param_grid=param_grid,
 ... scoring='accuracy',
 ... cv=10,
 ... n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
0.984615384615
print(gs.best_params_)


##########################################################################################################################

gs1 = GridSearchCV(estimator=pipe1, param_grid=param_grid1, scoring='accuracy', cv=10, n_jobs=-1)
gs1.fit(X_train, y_train)
print("\n\n\n")
print("\ngs1.best_score_:")
print(gs1.best_score_)
print("\ngs1.best_params_:")
print(gs1.best_params_)
print("\ngs1.best_estimator_:")
print(gs1.best_estimator_)
print("\n\n\n")

##########################################################################################################################
# find best model
###########################################################################################################################

# set best_model to the best_estimator_ from GridSearchCV that has the best score 
# Compare the best scores from each grid search
best_score1 = gs1.best_score_

# Find the index of the grid search with the highest score
best_index = np.argmax([best_score1, 0,])

# Set best_model to the best estimator from the grid search with the highest score
if best_index == 0:
    best_model = gs1.best_estimator_
    best_params = gs1.best_params_
# Make predictions on the test data
y_pred_train = best_model.predict(X_train)
y_pred = best_model.predict(X_test)

# Calculate and print the accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n\n\n\nBest Model Train Accuracy: {train_accuracy:.2f}")
print(f"Best Model Test Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}\n")
print(f"Recall: {recall:.2f}\n")
print(f"F1: {f1:.2f}\n")
print("best_model")
print(best_model)
print("best_params:")
print(best_params)

##########################################################################################################################
# TODO: plot decision regions
###########################################################################################################################


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


X_combined = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train,y_test))

print(X_combined.shape)
print(y_combined.shape)
print(X_combined)

# plot_decision_regions(X=X_combined, y=y_combined, classifier=gs1_std, test_idx=range(y_train.size, y_train.size + y_test.size))
plot_decision_regions(X=X_combined, y=y_combined, classifier=best_model, test_idx=None, resolution=0.01)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('Part1/results/decision_regions.png')
plt.close()
'''