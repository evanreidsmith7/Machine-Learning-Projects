#************************************************************************************
# Evan Smith
# ML – HW#3
# Filename: hw3-perceptron.py
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
from sklearn.linear_model import Perceptron
import seaborn as sns
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
##########################################################################################################################
# PIPELINES
###########################################################################################################################
# Create a pipeline
pipe1 = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', PCA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', Perceptron())  # Change the classifier to Perceptron
])

pipe2 = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', LDA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', Perceptron())
])

pipe3 = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', PCA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', Perceptron())
])

pipe4 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('reduce_dim', PCA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', Perceptron())
])

pipe5 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('reduce_dim', LDA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', Perceptron())
])

pipe6 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('reduce_dim', PCA()),  # You can change this to LDA or other dimensionality reduction techniques
    ('classifier', Perceptron())
])

param_grid = {
    'reduce_dim__n_components': [1, 2, 3],  # Number of components for PCA
    'classifier__penalty': [None, 'l2', 'l1'],  # Regularization penalty (L2, L1, or None)
    'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],  # Regularization strength (alpha)
    'classifier__max_iter': [100, 250],  # Maximum number of iterations
    'classifier__tol': [1e-3, 1e-4, 1e-5],  # Tolerance for stopping criterion
}
speed_grid = {
    'reduce_dim__n_components': [1],  # Number of components for PCA
    'classifier__penalty': [None,'l1'],  # Regularization penalty (L2, L1, or None)
    'classifier__alpha': [0.1],  # Regularization strength (alpha)
    'classifier__max_iter': [100],  # Maximum number of iterations
    'classifier__tol': [1e-3],  # Tolerance for stopping criterion
}

##########################################################################################################################
# GridSearchCV
'''
gs = GridSearchCV(estimator=pipe_svc,
 ... param_grid=param_grid,
 ... scoring='accuracy',
 ... cv=10,
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

gs2 = GridSearchCV(estimator=pipe2, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs2.fit(X_train, y_train)
print("\n\n\n")
print("\ngs2.best_score_:")
print(gs2.best_score_)
print("\ngs2.best_params_:")
print(gs2.best_params_)
print("\ngs2.best_estimator_:")
print(gs2.best_estimator_)
print("\n\n\n")

gs3 = GridSearchCV(estimator=pipe3, param_grid=speed_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs3.fit(X_train, y_train)
print("\n\n\n")
print("\ngs3.best_score_:")
print(gs3.best_score_)
print("\ngs3.best_params_:")
print(gs3.best_params_)
print("\ngs3.best_estimator_:")
print(gs3.best_estimator_)
print("\n\n\n")

gs4 = GridSearchCV(estimator=pipe4, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs4.fit(X_train, y_train)
print("\n\n\n")
print("\ngs4.best_score_:")
print(gs4.best_score_)
print("\ngs4.best_params_:")
print(gs4.best_params_)
print("\ngs4.best_estimator_:")
print(gs4.best_estimator_)
print("\n\n\n")

gs5 = GridSearchCV(estimator=pipe5, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs5.fit(X_train, y_train)
print("\n\n\n")
print("\ngs5.best_score_:")
print(gs5.best_score_)
print("\ngs5.best_params_:")
print(gs5.best_params_)
print("\ngs5.best_estimator_:")
print(gs5.best_estimator_)
print("\n\n\n")

gs6 = GridSearchCV(estimator=pipe6, param_grid=speed_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs6.fit(X_train, y_train)
print("\n\n\n")
print("\ngs6.best_score_:")
print(gs6.best_score_)
print("\ngs6.best_params_:")
print(gs6.best_params_)
print("\ngs6.best_estimator_:")
print(gs6.best_estimator_)
print("\n\n\n")

##########################################################################################################################
# find best model
###########################################################################################################################

# set best_model to the best_estimator_ from GridSearchCV that has the best score 
# Compare the best scores from each grid search
best_score1 = gs1.best_score_
best_score2 = gs2.best_score_
best_score3 = gs3.best_score_
best_score4 = gs4.best_score_
best_score5 = gs5.best_score_
best_score6 = gs6.best_score_

# Find the index of the grid search with the highest score
best_index = np.argmax([best_score1, best_score2, best_score3, best_score4, best_score5, best_score6])

# Set best_model to the best estimator from the grid search with the highest score
if best_index == 0:
    y_prob = gs1.best_estimator_.decision_function(X_test)
    best_model = gs1.best_estimator_
    best_params = gs1.best_params_
    best_model_dimreduce = gs1.best_estimator_['reduce_dim']
    best_model_scaler = gs1.best_estimator_['scaler']
    best_model_dimreduce_components = gs1.best_estimator_['reduce_dim'].n_components
    plotmodel = gs1.best_estimator_['classifier']
elif best_index == 1:
    y_prob = gs2.best_estimator_.decision_function(X_test)
    best_model = gs2.best_estimator_
    best_params = gs2.best_params_
    best_model_dimreduce = gs2.best_estimator_['reduce_dim']
    best_model_scaler = gs2.best_estimator_['scaler']
    best_model_dimreduce_components = gs2.best_estimator_['reduce_dim'].n_components    
    plotmodel = gs2.best_estimator_['classifier']
elif best_index == 2:
    y_prob = gs3.best_estimator_.decision_function(X_test)(X_test)
    best_model = gs3.best_estimator_
    best_params = gs3.best_params_
    best_model_dimreduce = gs3.best_estimator_['reduce_dim']
    best_model_scaler = gs3.best_estimator_['scaler']
    best_model_dimreduce_components = gs3.best_estimator_['reduce_dim'].n_components
    plotmodel = gs3.best_estimator_['classifier']
elif best_index == 3:
    y_prob = gs4.best_estimator_.decision_function(X_test)(X_test)
    best_model = gs4.best_estimator_
    best_params = gs4.best_params_
    best_model_dimreduce = gs4.best_estimator_['reduce_dim']
    best_model_scaler = gs4.best_estimator_['scaler']
    best_model_dimreduce_components = gs4.best_estimator_['reduce_dim'].n_components
    plotmodel = gs4.best_estimator_['classifier']
elif best_index == 4:
    y_prob = gs5.best_estimator_.decision_function(X_test)(X_test)
    best_model = gs5.best_estimator_
    best_params = gs5.best_params_
    best_model_dimreduce = gs5.best_estimator_['reduce_dim']
    best_model_scaler = gs5.best_estimator_['scaler']
    best_model_dimreduce_components = gs5.best_estimator_['reduce_dim'].n_components
    plotmodel = gs5.best_estimator_['classifier']
else:
    y_prob = gs6.best_estimator_.decision_function(X_test)(X_test)
    best_model = gs6.best_estimator_
    best_params = gs6.best_params_
    best_model_dimreduce = gs6.best_estimator_['reduce_dim']
    best_model_scaler = gs6.best_estimator_['scaler']
    best_model_dimreduce_components = gs6.best_estimator_['reduce_dim'].n_components
    plotmodel = gs6.best_estimator_['classifier']
###############################################
##################DEBUG########################
###############################################
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
###############################################
##################DEBUG########################
###############################################



# Make predictions on the test data
y_pred_train = best_model.predict(X_train)
y_pred = best_model.predict(X_test)

# Calculate and print the accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("\n\n\n\n")
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
with open('Part2/results/results.txt', 'w') as file:
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
    plt.savefig("Part2/results/{}_Decision_Regions.png".format(name))
    plt.close()
# PLOT FUNC################################################################################################################
'''

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



'''

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

plotDecisionRegions("Perceptron", X=X_combined, y=y_combined, \
                          classifier=plotmodel, resolution=0.02)

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
    plt.savefig('Part2/results/learning_curve.png')
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
plt.savefig('Part2/results/training_confusion_matrix.png')
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
plt.savefig('Part2/results/test_confusion_matrix.png')
#plt.show()
plt.close()


##########################################################################################################################
#  ROC curve and AUC
###########################################################################################################################
# Create and save ROC AUC graph
#y_prob = best_model.decision_function(X_test)
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
plt.savefig('Part2/results/roc_auc.png')
#plt.show()
plt.close()