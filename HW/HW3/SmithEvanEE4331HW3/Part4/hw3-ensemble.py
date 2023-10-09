#************************************************************************************
# Evan Smith
# ML – HW#3
# Filename: hw3-ensemble.py
# Due: , 2023
#
# Objective:
#*************************************************************************************
print("Running hw3-ensemble.py\nimporting libraries...")
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
print("imports complete")
#from mlxtend.plotting import plot_decision_regions

##########################################################################################################################
# Data Preprocessing
print("\n\n\npreprocessing......\n\n\n")
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
num_subdirectories_to_traverse = 10  # Set to None to traverse all subdirectories
debugfile = 'Part4/results/'+str(num_subdirectories_to_traverse) + 'debugresults.txt'
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

print("\n\n\npreprocessing complete\n\n\n")
####GRID SEARCH FOR SUBMISSION############################################################################
####PERCEPTRON############################################################################
ppn_param_grid = {
    'reduce_dim__n_components': [1, 2, 3],  # Number of components for PCA
    'classifier__penalty': [None, 'l2', 'l1'],  # Regularization penalty (L2, L1, or None)
    'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],  # Regularization strength (alpha)
    'classifier__max_iter': [100, 250],  # Maximum number of iterations
    'classifier__tol': [1e-3, 1e-4, 1e-5],  # Tolerance for stopping criterion
}
ppn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', PCA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', Perceptron())  # Change the classifier to Perceptron
])
ppn_gs = GridSearchCV(estimator=ppn_pipe, param_grid=ppn_param_grid, scoring='accuracy', cv=5, n_jobs=-1)
ppn_gs.fit(X_train, y_train)
ppn = ppn_gs.best_estimator_
####PERCEPTRON############################################################################
####LOGISTIC REGRESSION###################################################################
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', LDA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', LogisticRegression())
])
lr_param_grid = {
    'reduce_dim__n_components': [1, 2, 3],  # Number of components for LDA
    'classifier__penalty': [None, 'l2'],  # Regularization penalty (L1 or L2)
    'classifier__C': [0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
    'classifier__solver': ['sag','lbfgs', 'saga'],  # Solver algorithms
    'classifier__max_iter': [100, 250, 500]  # Maximum number of iterations
}
lr_gs = GridSearchCV(estimator=lr_pipe, param_grid=lr_param_grid, scoring='accuracy', cv=5, n_jobs=-1)
lr_gs.fit(X_train, y_train)
lr = lr_gs.best_estimator_
#####LOGISTIC REGRESSION###################################################################
####GRID SEARCH FOR SUBMISSION############################################################################



####REGULAR PIPE FOR TESTING##############################################################################
'''
####PERCEPTRON############################################################################
ppn = Pipeline(steps=[('scaler', StandardScaler()),
                ('reduce_dim', LDA(n_components=3)),
                ('classifier',
                 Perceptron(alpha=0.001, max_iter=100, penalty='l1',
                            tol=0.0001))])
####PERCEPTRON############################################################################
####LOGISTIC REGRESSION###################################################################
lr = Pipeline(steps=[('scaler', MinMaxScaler()),
                ('reduce_dim', LDA(n_components=3)),
                ('classifier',
                 LogisticRegression(C=0.01, penalty='l1', solver='saga'))])
####LOGISTIC REGRESSION###################################################################
'''
####DTREE#################################################################################
dtree = Pipeline(steps=[('classifier',
                 DecisionTreeClassifier(criterion='entropy',
                                        max_features='log2'))])
####DTREE#################################################################################
####REGULAR PIPE FOR TESTING##############################################################################





####my_clf#################################################################################

clf_labels = ['Logistic Regression', 'Perceptron', 'Decision Tree']
ensemble = VotingClassifier(estimators=[('lr', lr), ('ppn', ppn), ('dtree', dtree)], voting='hard', n_jobs=-1)
# Define a parameter grid for the weights
my_clf_param_grid = {
    'weights': [
        [1, 1, 1],  # Equal weights
        [2, 1, 3],  # Assigning higher weight to the lr
        [1, 1, 3],  # d tree bias
        [2, 1, 4]
        # Add more weight combinations to explore
    ]
}
# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=ensemble, param_grid=my_clf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
my_clf = grid_search.best_estimator_

clf_labels += ['my_clf']
all_clf = [lr, ppn, dtree, my_clf]

my_clf.fit(X_train, y_train)
####my_clf#################################################################################

y_pred_train = my_clf.predict(X_train)
y_pred = my_clf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

##########################################################################################################################
# txt file
###########################################################################################################################

# print to a txt file
with open('Part4/results/results.txt', 'w') as file:
    file.write(f"\nBest Model Train Accuracy: {train_accuracy:.2f}")
    file.write(f"\nBest Model Test Accuracy: {accuracy:.2f}")
    file.write(f"\nPrecision: {precision:.2f}\n")
    file.write(f"\nRecall: {recall:.2f}\n")
    file.write(f"\nF1: {f1:.2f}\n")

with open(debugfile, 'w') as file:
    file.write(f"\nBest Model Train Accuracy: {train_accuracy:.2f}")
    file.write(f"\nBest Model Test Accuracy: {accuracy:.2f}")
    file.write(f"\nPrecision: {precision:.2f}\n")
    file.write(f"\nRecall: {recall:.2f}\n")
    file.write(f"\nF1: {f1:.2f}\n")
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf,
        X=X_train,
        y=y_train,
        cv=5,
        scoring='accuracy',
        n_jobs=-1)

        print("\n\naccuracy: %0.2f (+/- %0.2f) [%s]"
        % (scores.mean(), scores.std(), label))

        file.write("\naccuracy: %0.2f (+/- %0.2f) [%s]"
        % (scores.mean(), scores.std(), label))

##########################################################################################################################
# TODO: plot decision regions
###########################################################################################################################
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
    plt.savefig("Part4/results/{}_Decision_Regions.png".format(name))
    plt.close()
# PLOT FUNC################################################################################################################
X_train_plt = X_train.values[:, [0, 1]]
X_test_plt = X_test.values[:, [0, 1]]
plotmodel = grid_search.best_estimator_
plotmodel.fit(X_train_plt, y_train)
print(X_train_plt.shape)

X_combined = np.vstack((X_train_plt, X_test_plt))
y_combined = np.hstack((y_train, y_test))

plotDecisionRegions("ensemble", X_combined, y_combined, classifier=plotmodel)
'''
for clf, label in zip(all_clf, clf_labels):
    plotDecisionRegions(label, X_train.values, y_train, classifier=clf)
'''

'''
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
        print("accuracy: %0.2f (+/- %0.2f) [%s]"
            % (scores.mean(), scores.std(), label))
        
        file.write("accuracy: %0.2f (+/- %0.2f) [%s]"
            % (scores.mean(), scores.std(), label))
    

with open('Part4/results/debugresults.txt', 'w') as file:
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf,
                                X=X_train,
                                y=y_train,
                                cv=5,
                                scoring='accuracy',
                                n_jobs=-1)
        
        print("accuracy: %0.2f (+/- %0.2f) [%s]"
            % (scores.mean(), scores.std(), label))
        
        file.write("accuracy: %0.2f (+/- %0.2f) [%s]"
            % (scores.mean(), scores.std(), label))
'''
