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
        combined_data = pd.concat([combined_data, data], ignore_index=True)

# add the labels to the dataframe
combined_data['label'] = 'Corridor'
combined_data.to_csv("datasets\combined_data_Corridor_rm155_71_loc0000.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Lab139_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Lab139_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data = pd.concat([combined_data, data], ignore_index=True)

combined_data['label'] = 'Lab139'
combined_data.to_csv("datasets\combined_data_Lab139_71_loc0000.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Main_Lobby71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Main_Lobby71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data = pd.concat([combined_data, data], ignore_index=True)

combined_data['label'] = 'Main_Lobby'
combined_data.to_csv("datasets\combined_data_Main_Lobby71_loc0000.csv", index=False)

# Loop through each file in the folder and read the data into a dataframe
for filename in os.listdir(Sport_Hall_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Sport_Hall_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data = pd.concat([combined_data, data], ignore_index=True)

combined_data['label'] = 'Sport_Hall'
combined_data.to_csv("datasets\combined_data_Sport_Hall_71_loc0000.csv", index=False)




#print(df.head())

def extract_label(file_name):
    if 'Corridor_rm155_71_loc0000_path' in file_name:
        return 'Corridor'
    elif 'Lab139_71_loc0000_path' in file_name:
        return 'Lab'
    elif 'Main_Lobby71_loc0000_path' in file_name:
        return 'Main Lobby'
    elif 'Sport_Hall_71_loc0000_path' in file_name:
        return "Sport Hall"
    else:
        return None
'''
print(df.head())
# Open the CSV file with semicolon as the delimiter
with open(loc0000csvfile, 'r') as file:
    # Create a CSV reader object with semicolon delimiter
    csv_reader = csv.reader(file, delimiter=';')
    
    # Iterate through the rows in the CSV file
    for row in csv_reader:
        # Each 'row' is a list of values in that row
        # Access the values by their index
        # For example, row[0] gets the first value in the row
        print(row)
'''