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
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, auc
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#from mlxtend.plotting import plot_decision_regions



# PLOT FUNC################################################################################################################
def plotDecisionRegions(name,X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
        y=X[y == cl, 1],
        alpha=0.8,
        c=colors[idx],
        marker=markers[idx],
        label=cl,
        edgecolor='black')


    plt.title("{} Decision Regions".format(name))
    plt.legend()
    plt.savefig("Part1/TestScripts/Results/{}_Decision_Regions.png".format(name))
    plt.close()
# PLOT FUNC################################################################################################################
###########################################################################################################################
###########################################################################################################################

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
##########################################################################################################################
# PIPELINES
###########################################################################################################################

pipe1 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('reduce_dim', LDA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression(C=0.01, penalty='l1', solver='saga'))
])
param_grid = {
    'reduce_dim__n_components': [3],  # Number of components for 
    'classifier__penalty': [None],  # Regularization penalty (L1 or L2)
    'classifier__C': [0.01, 0.1],  # Inverse of regularization strength
    'classifier__solver': ['sag'],  # Solver algorithms
    'classifier__max_iter': [250]  # Maximum number of iterations
}
gs = GridSearchCV(estimator=pipe1, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
print("\n\n\n")
print("\ngs1.best_score_:")
print(gs.best_score_)
print("\ngs1.best_params_:")
print(gs.best_params_)
print("\ngs1.best_estimator_:")
print(gs.best_estimator_)
print("\n\n\n")

###########################################################################################################################
# train the pipe
###########################################################################################################################
best_model = gs.best_estimator_
best_model_dimreduce = gs.best_estimator_['reduce_dim']
best_model_scaler = gs.best_estimator_['scaler']
best_model_dimreduce_components = gs.best_estimator_['reduce_dim'].n_components
print("\n\n\n")
print("\nbest_model:")
print(best_model)
print("\nbest_model_dimreduce:")
print(best_model_dimreduce)
print("\nbest_model_scaler:")
print(best_model_scaler)
print("\nbest_model_dimreduce_components:")
print(best_model_dimreduce_components)
print("\n\n\n")


best_model.fit(X_train, y_train)
y_pred_train = best_model.predict(X_train)
y_pred = best_model.predict(X_test)

###########################################################################################################################
###########################################################################################################################
# PLOT CALL################################################################################################################
###################################################################################################PCA SC##################
############################################################################
# Plot the decision region
############################################################################

X_train_plt = best_model_scaler.fit_transform(X_train)
X_test_plt = best_model_scaler.transform(X_test)
if type(best_model_dimreduce) == PCA():
    pca = PCA(n_components=2)
    X_train_plt = pca.fit_transform(X_train_plt)
    X_test_plt = pca.transform(X_test_plt)
else:
    lda = LDA(n_components=2)
    X_train_plt = lda.fit_transform(X_train_plt, y_train)
    X_test_plt = lda.transform(X_test_plt)

plotmodel = best_model['classifier']
plotmodel.fit(X_train_plt[:, :2], y_train)
print(X_train_plt.shape)

X_combined = np.vstack((X_train_plt[:, :2], X_test_plt[:, :2]))
y_combined = np.hstack((y_train, y_test))

plotDecisionRegions("Logistic Regression", X=X_combined, y=y_combined, \
                          classifier=plotmodel, resolution=0.02)

############################################################################

# Calculate and print the accuracy
train_accuracy5 = accuracy_score(y_train, y_pred_train)
accuracy5 = accuracy_score(y_test, y_pred)
precision5 = precision_score(y_test, y_pred, average='weighted')
recall5 = recall_score(y_test, y_pred, average='weighted')
f15 = f1_score(y_test, y_pred, average='weighted')

print(f"\n\n\n\n5 Train Accuracy: {train_accuracy5:.2f}")
print(f"5 Test Accuracy: {accuracy5:.2f}")
print(f"5 Precision: {precision5:.2f}\n")
print(f"5 Recall: {recall5:.2f}\n")
print(f"5 F1: {f15:.2f}\n")

##########################################################################################################################
# txt file
###########################################################################################################################

# print to a txt file
with open('Part1/TestScripts/Results/210plotTestresults.txt', 'w') as file:

   file.write(f"\n\n\n\n5 Train Accuracy: {train_accuracy5:.2f}")
   file.write(f"5 Test Accuracy: {accuracy5:.2f}")
   file.write(f"5 Precision: {precision5:.2f}\n")
   file.write(f"5 Recall: {recall5:.2f}\n")
   file.write(f"5 F1: {f15:.2f}\n")
##########################################################################################################################
# END
###########################################################################################################################