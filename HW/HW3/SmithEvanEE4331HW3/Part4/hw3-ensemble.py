#************************************************************************************
# Evan Smith
# ML – HW#3
# Filename: hw3-ensemble.py
# Due: , 2023
#
# Objective:
#*************************************************************************************
import os
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Perceptron
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#from mlxtend.plotting import plot_decision_regions

##########################################################################################################################
# Data Preprocessing
###########################################################################################################################
# Set the path to the main directory containing subdirectories with CSV files
main_directory = r'Datasets/Measurements_Upload/'



# Define the list of paths
paths = [
    r'Datasets/Measurements_Upload/Corridor_rm155_7.1',
    r'Datasets/Measurements_Upload/Lab139_7.1',
    r'Datasets/Measurements_Upload/Main_Lobby_7.1',
    r'Datasets/Measurements_Upload/Sport_Hall_7.1'
]

# Number of subdirectories to traverse (you can adjust this as needed)
num_subdirectories_to_traverse = 1  # Set to None to traverse all subdirectories

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


svc = Pipeline(steps=[('scaler', StandardScaler()),
                ('reduce_dim', LDA(n_components=3)),
                ('classifier',
                 SVC(C=1, gamma=0.1, kernel='rbf'))])


lr = Pipeline(steps=[('scaler', MinMaxScaler()),
                ('reduce_dim', LDA(n_components=3)),
                ('classifier',
                 LogisticRegression(C=0.01, penalty=None, solver='saga'))])
ppn = Pipeline(steps=[('scaler', StandardScaler()),
                ('reduce_dim', LDA(n_components=3)),
                ('classifier',
                 Perceptron(alpha=0.001, max_iter=100, penalty='l1',
                            tol=0.0001))])
dtree = Pipeline(steps=[('classifier',
                 DecisionTreeClassifier(criterion='entropy',
                                        max_features='log2'))])


clf_labels = ['Logistic Regression', 'Perceptron', 'Decision Tree', 'SVM']
my_clf = VotingClassifier(estimators=[('lr', lr), ('ppn', ppn), ('dtree', dtree), ('svc', svc)], voting='hard')
clf_labels += ['Majority voting - ArgMax']
all_clf = [lr, ppn, dtree, my_clf]

with open('Part4/results/results.txt', 'w') as file:
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf,
                                X=X_train,
                                y=y_train,
                                cv=5,
                                scoring='roc_auc',
                                n_jobs=-1)
        
        print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
            % (scores.mean(), scores.std(), label))
        
        file.write("ROC AUC: %0.2f (+/- %0.2f) [%s]"
            % (scores.mean(), scores.std(), label))
    

with open('Part4/results/debugresults.txt', 'w') as file:
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf,
                                X=X_train,
                                y=y_train,
                                cv=50,
                                scoring='accuracy',
                                n_jobs=-1)
        
        print("accuracy: %0.2f (+/- %0.2f) [%s]"
            % (scores.mean(), scores.std(), label))
        
        file.write("accuracy: %0.2f (+/- %0.2f) [%s]"
            % (scores.mean(), scores.std(), label))

    
'''
mv_clf = VotingClassifier(estimators=[('lr', pipe1),
 ... ('dt', clf2), ('knn', pipe3)], voting='soft')
clf_labels += ['Majority voting - ArgMax']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
... scores = cross_val_score(estimator=clf,
... X=X_train,
... y=y_train,
... cv=10,
... scoring='roc_auc')
... print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
... % (scores.mean(), scores.std(), label))

'''