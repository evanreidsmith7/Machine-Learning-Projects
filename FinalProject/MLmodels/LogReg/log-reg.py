# Import libraries
print("Importing libraries...")
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import learning_curve
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
from sklearn.linear_model import LogisticRegression
import seaborn as sns

################################################################################################################################################
# Functions
################################################################################################################################################

# Function to load CSV files from a directory
def load_csv_files(directory):
    dataset = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            dataset = pd.concat([dataset, df], ignore_index=True)
    return dataset

################################################################################################################################################
# Data preprocessing
################################################################################################################################################
print("Data preprocessing...")

region1_path = r"Datasets/Sample Data/Region1"
region2_path = r"Datasets/Sample Data/Region2"
region3_path = r"Datasets/Sample Data/Region3"
region4_path = r"Datasets/Sample Data/Region4"
region5_path = r"Datasets/Sample Data/Region5"

region_paths = [
    r"Datasets/Sample Data/Region1",
    r"Datasets/Sample Data/Region2",
    r"Datasets/Sample Data/Region3",
    r"Datasets/Sample Data/Region4",
    r"Datasets/Sample Data/Region5",
]

# empty dataframe to store entire dataset
full_dataset = pd.DataFrame()

# Load CSV files from each region and concatenate them
for region_path in region_paths:
    region_dataset = load_csv_files(region_path)
    full_dataset = pd.concat([full_dataset, region_dataset], ignore_index=True)


# Handling missing values by dropping rows with missing values
full_dataset.dropna(inplace=True)

# Display the shape of the combined dataset
print("Shape of the combined dataset:", full_dataset.shape)
print("\n\n\n", full_dataset)


y = full_dataset["Region"]
# encode y
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("\n\n\n", y)



X = full_dataset.drop(["Region"], axis=1)
# Convert empty strings to NaN
X[X == ''] = np.nan

# Convert all values to float
X = X.astype(float)
print("\n\n\n", X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

##########################################################################################################################
# PIPELINES
###########################################################################################################################
print("Pipelines...")
# Create a pipeline
pipe1 = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', PCA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression())
])
param_grid1 = {
    'reduce_dim__n_components': [1, 2, 3],  # Number of components for PCA
    'classifier__penalty': [None, 'l2'],  # Regularization penalty (L1 or L2)
    'classifier__C': [0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
    'classifier__solver': ['sag','lbfgs', 'saga'],  # Solver algorithms
    'classifier__max_iter': [100, 250, 500]  # Maximum number of iterations
}


##########################################################################################################################
# Grid Search
##########################################################################################################################
print("Grid Search...")

gs1 = GridSearchCV(estimator=pipe1, param_grid=param_grid1, scoring='accuracy', cv=5, n_jobs=-1)
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
# create best model

best_model = gs1.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print("\n\n\n")
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Precision score:", precision_score(y_test, y_pred, average='weighted'))