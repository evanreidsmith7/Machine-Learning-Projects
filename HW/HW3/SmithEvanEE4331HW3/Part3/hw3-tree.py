#************************************************************************************
# Evan Smith
# ML – HW#3
# Filename: hw3-tree.py
# Due: , 2023
#
# Objective:
#*************************************************************************************
# Import libraries
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
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
#from mlxtend.plotting import plot_decision_regions

##########################################################################################################################
# Data Preprocessing
###########################################################################################################################
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
num_subdirectories_to_traverse = 10  # Set to None to traverse all subdirectories

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
print(X)
print(X.shape)
print(y_encoded)
print(y_encoded.shape)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

##########################################################################################################################
# PIPELINES
###########################################################################################################################
pipe1 = Pipeline([
 # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', DecisionTreeClassifier())
])

param_grid = {
    'classifier__criterion': ['gini', 'entropy'],  # Split criterion
    'classifier__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'classifier__min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'classifier__min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node
    'classifier__max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for split
}

##########################################################################################################################
# GridSearchCV
'''
gs = GridSearchCV(estimator=pipe_svc,
 ... param_grid=param_grid,
 ... scoring='accuracy',
 ... cv=5,
 ... n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
0.984615384615
print(gs.best_params_)
'''
##########################################################################################################################

gs1 = GridSearchCV(estimator=pipe1, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
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
y_prob = gs1.predict_proba(X_test)
best_score1 = gs1.best_score_
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
print("\n\n\n\n")
print(X.shape)
print(y_encoded.shape)
##########################################################################################################################
# txt file
###########################################################################################################################

# print to a txt file
with open('Part3/results/results.txt', 'w') as file:
    file.write(f"\nBest Model Train Accuracy: {train_accuracy:.2f}")
    file.write(f"\nBest Model Test Accuracy: {accuracy:.2f}")
    file.write(f"\nPrecision: {precision:.2f}\n")
    file.write(f"\nRecall: {recall:.2f}\n")
    file.write(f"\nF1: {f1:.2f}\n")
    file.write("\nbest_model:\n")
    file.write(str(best_model))
    file.write("\nbest_params:\n")
    file.write(str(best_params))


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


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
pca = PCA(n_components=2)
X_train_dim = pca.fit_transform(X_train_std)
X_test_dim = pca.transform(X_test_std)
plotmodel = best_model.fit(X_train_dim, y_train)

X_combined = np.vstack((X_train_dim,X_test_dim))
y_combined = np.hstack((y_train,y_test))

print(X_combined.shape)
print(y_combined.shape)
print(X_combined)
# plot_decision_regions(X=X_combined, y=y_combined, classifier=gs1_std, test_idx=range(y_train.size, y_train.size + y_test.size))
plot_decision_regions(X=X_combined, y=y_combined, classifier=plotmodel, test_idx=None, resolution=0.01)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('Part3/results/decision_regions.png')
plt.close()



##########################################################################################################################
# learning curves
###########################################################################################################################
# Create and save learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('Part3/results/learning_curve.png')
    #plt.show()
    plt.close()

plot_learning_curve(best_model, "Learning Curve", X_train, y_train, cv=5, n_jobs=-1)


##########################################################################################################################
# training confusion matrix
###########################################################################################################################
# Create a confusion matrix for y_train
confusion_train = confusion_matrix(y_train, y_pred_train)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_train, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix y_train")
plt.savefig('Part3/results/training_confusion_matrix.png')
#plt.show()
plt.close()

##########################################################################################################################
# test confusion matrix
###########################################################################################################################
# Create a confusion matrix for y_train
confusion_test = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_test, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix y_test")
plt.savefig('Part3/results/test_confusion_matrix.png')
#plt.show()
plt.close()


##########################################################################################################################
#  ROC curve and AUC
###########################################################################################################################
# Create and save ROC AUC graph
#y_prob = best_model.predict_proba(X_test)
n_classes = len(np.unique(y_test))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2,
             label='ROC curve (area = %0.2f) for class %d' % (roc_auc[i], i))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC')
plt.legend(loc="lower right")
plt.savefig('Part3/results/roc_auc.png')
#plt.show()
plt.close()