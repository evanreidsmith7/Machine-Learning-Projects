#************************************************************************************
# Evan Smith
# ML – HW#2
# Filename: hw2-svm.py
# Due: Sept. 20, 2023
#
# Objective:
'''
• Use a scikit-learn SVM model to classify all targets.
• Determine which features and parameters will provide the best outcomes using PCA or LDA
• From the dataset:
    o For each location, choose a single loc_number folder (use this one for all models).
    o Within the folder, combine all the CSV files into a single file for that location.
        § Label the data accordingly to the dataset information.
    o You can either keep it separate or join all samples to a large master CSV file.
• In your program, print out the training and test accuracy values and the best features values via
  the PCA or LDA to a text file from the best model experiment.
• Generate the t-SNE and UMAP images.
• Generate the training and testing Plot Decision image.
'''
#*************************************************************************************
# Import libraries
from umap import UMAP
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
# * labels *
# corridor_rm155_7.1
# Lab139_7.1
# Main_Lobby7.1
# Sport_Hall_7.1


###########################################################################################################################
# Data Preprocessing
###########################################################################################################################

# Set the path of each of the folders we want to excract from
Corridor_rm155_71_loc0000_path = 'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'
Lab139_71_loc0000_path = 'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'
Main_Lobby71_loc0000_path = 'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'
Sport_Hall_71_loc0000_path = 'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'
RESULTS_ACCURACY_LOCATION = 'HW\HW2\SmithEvanEE4331HW2\Part1\part1results.txt'
RESULTS_DIR = 'HW\HW2\SmithEvanEE4331HW2\Part1\results'
TSNE_LOCATION = 'HW\HW2\SmithEvanEE4331HW2\Part1\Results\_tsne.png'
UMAP_LOCATION = 'HW\HW2\SmithEvanEE4331HW2\Part1\Results\_umap.png'

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
combined_data1.to_csv("datasets\combined_data_Corridor_rm155_71_loc0001.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Lab139_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Lab139_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data2 = pd.concat([combined_data, data], ignore_index=True)

combined_data2['label'] = 'Lab139'
combined_data2.to_csv("datasets\combined_data_Lab139_71_loc0001.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Main_Lobby71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Main_Lobby71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data3 = pd.concat([combined_data, data], ignore_index=True)

combined_data3['label'] = 'Main_Lobby'
combined_data3.to_csv("datasets\combined_data_Main_Lobby71_loc0001.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Sport_Hall_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Sport_Hall_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data4 = pd.concat([combined_data, data], ignore_index=True)

combined_data4['label'] = 'Sport_Hall'
combined_data4.to_csv("datasets\combined_data_Sport_Hall_71_loc0001.csv", index=False)

# List of CSV files to combine
csv_files = [
    'datasets\combined_data_Corridor_rm155_71_loc0001.csv',
    'datasets\combined_data_Lab139_71_loc0001.csv',
    'datasets\combined_data_Main_Lobby71_loc0001.csv',
    'datasets\combined_data_Sport_Hall_71_loc0001.csv'
]

# combine all combined data into a single dataframe
data_frames = [combined_data1, combined_data2, combined_data3, combined_data4]

combined_data = pd.concat(data_frames, ignore_index=True)

combined_data.to_csv("combined_data_all.csv", index=False)

# drop the 5th collumn
combined_data = combined_data.drop(columns=[5])

# Assuming you have loaded your data into the combined_data DataFrame
# If not, you should load your data here

# Handling missing values by dropping rows with missing values
combined_data.dropna(inplace=True)

# Splitting into X (features) and y (labels)
X = combined_data.drop(columns=['label'])  # Assuming 'label' is the column containing your labels
X = X.astype(float)
#y = combined_data.drop(columns=[0,1,2,3,4])
y = combined_data['label']

# encode y
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# Print the shape of your data
print("Shape of X:", X.shape)
print("Unique labels:", np.unique(y))

# Print a sample of your data
print("Sample of X:")
print(X.head())

# Print labels after encoding
print("Encoded labels:", y_encoded)

# Print intermediary results
print("Combined Data Shape:", combined_data.shape)
print("Sample of Combined Data:")
print(combined_data.head())
###########################################################################################################################
# generate t-SNE and UMAP images
###########################################################################################################################
'''
# Initialize t-SNE
#tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_jobs=-1)
tsne = TSNE(n_components=2, n_jobs=-1)

# Run t-SNE and get the transformed 2D representation
X_tsne_2d = tsne.fit_transform(X)


# Scatter plot for each class label
plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(y_encoded)):
    plt.scatter(X_tsne_2d[y_encoded == i, 0], X_tsne_2d[y_encoded == i, 1], label=label)
plt.legend()
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('2D t-SNE on Dataset')
plt.savefig(TSNE_LOCATION)

'''

'''
# Initialize UMAP
umap_model = UMAP(n_neighbors=50, min_dist=1,n_components=2)
# Run UMAP and get the transformed 2D representation
X_umap_2d = umap_model.fit_transform(X)

# Scatter plot for each class label
for i, label in enumerate(np.unique(y_encoded)):
    plt.scatter(X_umap_2d[y_encoded == i, 0], X_umap_2d[y_encoded == i, 1], label=label)
#plt.legend()
plt.xlabel('UMAP feature 1')
plt.ylabel('UMAP feature 2')
plt.title('2D UMAP on Dataset')
plt.savefig(UMAP_LOCATION)


# Print the shape of your data
print("Shape of X:", X.shape)
print("Unique labels:", np.unique(y))

# Print a sample of your data
print("Sample of X:")
print(X.head())

# Print labels after encoding
print("Encoded labels:", y_encoded)

# Print parameters for t-SNE and UMAP plotting
print("t-SNE Plotting Parameters:")
#print("X_tsne_2d shape:", X_tsne_2d.shape)
print("Labels for t-SNE:", np.unique(y))

print("UMAP Plotting Parameters:")
#print("X_umap_2d shape:", X_umap_2d.shape)
print("Labels for UMAP:", np.unique(y))

# Print intermediary results
print("Combined Data Shape:", combined_data.shape)
print("Sample of Combined Data:")
print(combined_data.head())
'''

###########################################################################################################################
# standardize and normalize
###########################################################################################################################

# standardize X: X_std
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)
print(X_std)
# normalize X: X_norm
mms = MinMaxScaler()
X_norm = mms.fit_transform(X)

print("X_std shape:", X_std.shape)

###########################################################################################################################
# PCA
###########################################################################################################################
print("Before LDA:")
print("X_std shape:", X_std.shape)
print("Unique labels:", np.unique(y))

lda = LDA(n_components=2)

lda = lda.fit(X_std, y_encoded)
X_lda = lda.transform(X_std)

print("After LDA:")
print(X_lda)
print("X_lda shape:", X_lda.shape)
print("Explained variance ratio:", lda.explained_variance_ratio_)
###########################################################################################################################
# SVM
###########################################################################################################################

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lda, y_encoded, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

svm_lda_std = SVC(kernel='linear', C=.01, random_state=42)
svm_lda_std.fit(X_train, y_train)

print("SVM model fitted.")

y_pred_train_lda_std = svm_lda_std.predict(X_train)
y_pred_test_lda_std = svm_lda_std.predict(X_test)



# Print the accuracy of the model
print("Training Accuracy:", accuracy_score(y_train, y_pred_train_lda_std))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test_lda_std))

###########################################################################################################################
# plot decision regions function
###########################################################################################################################
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

	plt.scatter(X_test[:,0],X_test[:,1],facecolors='none', edgecolors='black', 
			    alpha=1.0,linewidths=1, marker='o', s=55, label='test set')
	
###########################################################################################################################
# End of PLOTTING DECISION CODE
###########################################################################################################################
'''
# Combine all training and test data to single object variables
X_combined_std=np.vstack((X_train,X_test_norm))
y_combined=np.hstack((y_train_norm,y_test_norm))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm_lda_std)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.savefig('HW\HW2\SmithEvanEE4331HW2\Part1\Results\svm_lda_norm_decregions.png')
'''