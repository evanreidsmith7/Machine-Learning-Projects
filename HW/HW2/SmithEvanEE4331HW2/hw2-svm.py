#************************************************************************************
# Evan Smith
# ML – HW#2
# Filename: hw2-svm.py
# Due: Sept. 14, 2023
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
import pandas as pd
import numpy as np
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
# * labels *
# corridor_rm155_7.1
# Lab139_7.1
# Main_Lobby7.1
# Sport_Hall_7.1

# Set the path of each of the folders we want to excract from
Corridor_rm155_71_loc0000_path = 'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'
Lab139_71_loc0000_path = 'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'
Main_Lobby71_loc0000_path = 'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'
Sport_Hall_71_loc0000_path = 'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'

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
combined_data1.to_csv("datasets\combined_data_Corridor_rm155_71_loc0000.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Lab139_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Lab139_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data2 = pd.concat([combined_data, data], ignore_index=True)

combined_data2['label'] = 'Lab139'
combined_data2.to_csv("datasets\combined_data_Lab139_71_loc0000.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Main_Lobby71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Main_Lobby71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data3 = pd.concat([combined_data, data], ignore_index=True)

combined_data3['label'] = 'Main_Lobby'
combined_data3.to_csv("datasets\combined_data_Main_Lobby71_loc0000.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Sport_Hall_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Sport_Hall_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data4 = pd.concat([combined_data, data], ignore_index=True)

combined_data4['label'] = 'Sport_Hall'
combined_data4.to_csv("datasets\combined_data_Sport_Hall_71_loc0000.csv", index=False)

# List of CSV files to combine
csv_files = [
    'datasets\combined_data_Corridor_rm155_71_loc0000.csv',
    'datasets\combined_data_Lab139_71_loc0000.csv',
    'datasets\combined_data_Main_Lobby71_loc0000.csv',
    'datasets\combined_data_Sport_Hall_71_loc0000.csv'
]

# combine all combined data into a single dataframe
data_frames = [combined_data1, combined_data2, combined_data3, combined_data4]

combined_data = pd.concat(data_frames, ignore_index=True)

combined_data.to_csv("combined_data_all.csv", index=False)

# drop the 5th collumn
combined_data = combined_data.drop(columns=[5])
print("dropping 5th col")
print(combined_data)

# Assuming you have loaded your data into the combined_data DataFrame
# If not, you should load your data here

# Handling missing values by dropping rows with missing values
combined_data.dropna(inplace=True)
print("dropping na")
print(combined_data)

# Splitting into X (features) and y (labels)
X = combined_data.drop(columns=['label'])  # Assuming 'label' is the column containing your labels
print("X")
print(X)

y = combined_data['label']
print("y")
print(y)


'''

# Splitting into X (features) and y (labels)
X = combined_data.drop(columns=['label'])  # Assuming 'label' is the column containing your labels
X = pd.get_dummies(X)  # One-hot encoding categorical columns
y = combined_data['label']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating an SVM model and training it
svm_model = SVC(kernel='linear', C=1.0, random_state=42)  # You can adjust the kernel and C value as needed
svm_model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = svm_model.predict(X_test)

# Evaluating the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
'''