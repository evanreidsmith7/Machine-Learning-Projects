#************************************************************************************
# Evan Smith
# ML – HW#3
# Filename: hw3-ensemble.py
# Due: , 2023
#
# Objective:
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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
#from mlxtend.plotting import plot_decision_regions

##########################################################################################################################
# plot func
###########################################################################################################################
def plot_decision_regions(X, y, clf, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^')  # Add a marker for the fourth class
    colors = ('red', 'blue', 'lightgreen', 'purple')  # Add a color for the fourth class
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lez = LabelEncoder()
    Z = lez.fit_transform(Z)
    Z = Z.reshape(xx1.shape)
    Z = Z.astype(float)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    """ Here is the explanation for the code above:
1. First, we determined the minimum and maximum values for the two features and used those feature vectors to create a pair of grid arrays xx1 and xx2 via the NumPy meshgrid function.
2. Since we trained our logistic regression classifier on two feature dimensions, we need to flatten the grid arrays and create a matrix that has the same number of columns as the Iris training dataset so that we can use the predict method to predict the class labels Z of """

    # Plot all the samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, color=cmap(idx), marker=markers[idx], label=cl)

    # Highlight test samples if test_idx is provided
    if test_idx is not None:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')

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
X = X.astype(float)
#y = combined_data.drop(columns=[0,1,2,3,4])
y = combined_data['label']

# encode y
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(y_encoded)
print(X.head())