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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, auc
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#from mlxtend.plotting import plot_decision_regions

##########################################################################################################################
# Data Preprocessing
###########################################################################################################################

main_directory = r'Datasets/Measurements_Upload/'



# Define the list of paths
paths = [
    r'Datasets/Measurements_Upload/Corridor_rm155_7.1',
    r'Datasets/Measurements_Upload/Lab139_7.1',
    r'Datasets/Measurements_Upload/Main_Lobby_7.1',
    r'Datasets/Measurements_Upload/Sport_Hall_7.1'
]

# Number of subdirectories to traverse (you can adjust this as needed)
num_subdirectories_to_traverse = 100  # Set to None to traverse all subdirectories

# Function to process a directory and return combined data
def process_directory(directory_path, label):
    combined_data = pd.DataFrame()
    
    for root, dirs, files in os.walk(directory_path):
        for subdir in dirs[:num_subdirectories_to_traverse]:
            subdir_path = os.path.join(root, subdir)
            
            for filename in os.listdir(subdir_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(subdir_path, filename)
                    data = pd.read_csv(file_path)
                    data = data['# Version 1.00'].str.split(';', expand=True)
                    data = data.drop([0, 1])
                    combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    combined_data['label'] = label
    return combined_data

# Create a list of dataframes for each path
data_frames = []

for path in paths:
    label = os.path.basename(path)
    data_frame = process_directory(path, label)
    data_frames.append(data_frame)

# Concatenate all dataframes into one
combined_data = pd.concat(data_frames, ignore_index=True)


#combined_data.to_csv("Datasets/combined_data_all.csv", index=False)

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
##########################################################################################################################
# PIPELINES
###########################################################################################################################
# Create a pipeline
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

pipe2 = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', LDA(n_components=1)),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression(C=0.01, penalty='l1', solver='saga'))
])

pipe4 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('reduce_dim', PCA(n_components=1)),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression(C=0.01, penalty='l1', solver='saga'))
])

pipe5 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('reduce_dim', LDA(n_components=1)),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression(C=0.01, penalty='l1', solver='saga'))
])


###########################################################################################################################
# train the pipes
###########################################################################################################################
'''
scores = cross_val_score(estimator=pipe_lr,
... X=X_train,
... y=y_train,
... cv=10,
... n_jobs=-1) #use all available cores in the machine
'''
###########################################################################################################################
# pipe 2
###########################################################################################################################
pipe2.fit(X_train, y_train)
y_pred_train2 = pipe2.predict(X_train)
y_pred2 = pipe2.predict(X_test)

# Calculate and print the accuracy
train_accuracy2 = accuracy_score(y_train, y_pred_train2)
accuracy2 = accuracy_score(y_test, y_pred2)
precision2 = precision_score(y_test, y_pred2, average='weighted')
recall2 = recall_score(y_test, y_pred2, average='weighted')
f12 = f1_score(y_test, y_pred2, average='weighted')

###########################################################################################################################
#  pipe 4
###########################################################################################################################
pipe4.fit(X_train, y_train)
y_pred_train4 = pipe2.predict(X_train)
y_pred4 = pipe2.predict(X_test)

# Calculate and print the accuracy
train_accuracy4 = accuracy_score(y_train, y_pred_train4)
accuracy4 = accuracy_score(y_test, y_pred4)
precision4 = precision_score(y_test, y_pred4, average='weighted')
recall4 = recall_score(y_test, y_pred4, average='weighted')
f14 = f1_score(y_test, y_pred4, average='weighted')

###########################################################################################################################
#  pipe 5
###########################################################################################################################
pipe5.fit(X_train, y_train)
y_pred_train5 = pipe5.predict(X_train)
y_pred5 = pipe5.predict(X_test)

# Calculate and print the accuracy
train_accuracy5 = accuracy_score(y_train, y_pred_train5)
accuracy5 = accuracy_score(y_test, y_pred5)
precision5 = precision_score(y_test, y_pred5, average='weighted')
recall5 = recall_score(y_test, y_pred5, average='weighted')
f15 = f1_score(y_test, y_pred5, average='weighted')





print(f"\n\n\n\n2 Train Accuracy: {train_accuracy2:.2f}")
print(f"2 Test Accuracy: {accuracy2:.2f}")
print(f"2 Precision: {precision2:.2f}\n")
print(f"2 Recall: {recall2:.2f}\n")
print(f"2 F1: {f12:.2f}\n")

print(f"\n\n\n\n4 Train Accuracy: {train_accuracy4:.2f}")
print(f"4 Test Accuracy: {accuracy4:.2f}")
print(f"4 Precision: {precision4:.2f}\n")
print(f"4 Recall: {recall4:.2f}\n")
print(f"4 F1: {f14:.2f}\n")

print(f"\n\n\n\n2 Train Accuracy: {train_accuracy5:.2f}")
print(f"5 Test Accuracy: {accuracy5:.2f}")
print(f"5 Precision: {precision5:.2f}\n")
print(f"5 Recall: {recall5:.2f}\n")
print(f"5 F1: {f15:.2f}\n")

##########################################################################################################################
# txt file
###########################################################################################################################

# print to a txt file
with open('Part1/results/tuning/top3pipes100f.txt', 'w') as file:
   file.write(f"\n\n\n\n2 Train Accuracy: {train_accuracy2:.2f}")
   file.write(f"2 Test Accuracy: {accuracy2:.2f}")
   file.write(f"2 Precision: {precision2:.2f}\n")
   file.write(f"2 Recall: {recall2:.2f}\n")
   file.write(f"2 F1: {f12:.2f}\n")

   file.write(f"\n\n\n\n4 Train Accuracy: {train_accuracy4:.2f}")
   file.write(f"4 Test Accuracy: {accuracy4:.2f}")
   file.write(f"4 Precision: {precision4:.2f}\n")
   file.write(f"4 Recall: {recall4:.2f}\n")
   file.write(f"4 F1: {f14:.2f}\n")

   file.write(f"\n\n\n\n2 Train Accuracy: {train_accuracy5:.2f}")
   file.write(f"5 Test Accuracy: {accuracy5:.2f}")
   file.write(f"5 Precision: {precision5:.2f}\n")
   file.write(f"5 Recall: {recall5:.2f}\n")
   file.write(f"5 F1: {f15:.2f}\n")
