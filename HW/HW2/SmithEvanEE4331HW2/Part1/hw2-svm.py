#************************************************************************************
# Evan Smith
# ML – HW#2
# Filename: hw2-svm.py
# Due: Sept. 20, 2023
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
import os
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
Corridor_rm155_71_loc0000_path = r'datasets\Measurements_Upload\Measurements_Upload\Corridor_rm155_7.1\Loc_0000'
Lab139_71_loc0000_path =         r'datasets\Measurements_Upload\Measurements_Upload\Lab139_7.1\Loc_0000'
Main_Lobby71_loc0000_path =      r'datasets\Measurements_Upload\Measurements_Upload\Main_Lobby_7.1\Loc_0000'
Sport_Hall_71_loc0000_path =     r'datasets\Measurements_Upload\Measurements_Upload\Sport_Hall_7.1\Loc_0000'

TEXT_FILE_LOCATION = r'HW\HW2\SmithEvanEE4331HW2\Part1\Results\part1results.txt'
TSNE_LOCATION =      r'HW\HW2\SmithEvanEE4331HW2\Part1\Results\_tsne.png'
UMAP_LOCATION =      r'HW\HW2\SmithEvanEE4331HW2\Part1\Results\_umap.png'
PLOT_LOCATION =      r'HW\HW2\SmithEvanEE4331HW2\Part1\Results\_decisionregions.png'

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
'''
###########################################################################################################################
# generate t-SNE and UMAP images
###########################################################################################################################

#
# Initialize t-SNE
#tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_jobs=-1)
tsne = TSNE(n_components=2, n_jobs=-1)

# Run t-SNE and get the transformed 2D representation
X_tsne_2d = tsne.fit_transform(X)


# Scatter plot for each class label
plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(y_encoded)):
    plt.scatter(X_tsne_2d[y_encoded == i, 0], X_tsne_2d[y_encoded == i, 1], label=label)
plt.legend()
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('2D t-SNE on Dataset')
plt.savefig(TSNE_LOCATION)




# Initialize UMAP
umap_model = UMAP(n_neighbors=50, min_dist=1,n_components=2)
# Run UMAP and get the transformed 2D representation
X_umap_2d = umap_model.fit_transform(X)

# Scatter plot for each class label
for i, label in enumerate(np.unique(y_encoded)):
    plt.scatter(X_umap_2d[y_encoded == i, 0], X_umap_2d[y_encoded == i, 1], label=label)
#plt.legend()
plt.xlabel('UMAP feature 1')
plt.ylabel('UMAP feature 2')
plt.title('2D UMAP on Dataset')
plt.savefig(UMAP_LOCATION)

'''
###########################################################################################################################
# standardize and normalize
###########################################################################################################################

# standardize X: X_std
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)
# normalize X: X_norm
mms = MinMaxScaler()
X_norm = mms.fit_transform(X)

# Splitting into training and testing sets for std
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_std, y, test_size=0.2, random_state=42)

# Splitting into training and testing sets for norm
X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_norm, y, test_size=0.2, random_state=42)

###########################################################################################################################
# STD EXPERIMENT
########################################################################################################################### 

param_grid = {
    'C': [0.1, 1, 10, 50],
    'kernel': ['linear', 'rbf'],
}
# Define a list of values to iterate through for n_components
n_components_values = [2, 3]  # You can customize this list

# Initialize variables to track the best combination
best_std_technique = None
best_std_n_components = None
best_std_accuracy = 0.0  # Initialize with a low value

# Experiment with PCA, LDA, and KPCA
for n_components in n_components_values:
    for technique in ['pca', 'lda', 'kpca']:
        if technique == 'pca':
            pca = PCA(n_components=n_components)
            X_train_reduced = pca.fit_transform(X_train_std)
            X_test_reduced = pca.transform(X_test_std)
        elif technique == 'lda':
            lda = LDA(n_components=n_components, solver='eigen')
            X_train_reduced = lda.fit_transform(X_train_std, y_train_std)
            X_test_reduced = lda.transform(X_test_std)
        elif technique == 'kpca':
            kpca = KernelPCA(kernel='rbf', n_components=n_components)
            X_train_reduced = kpca.fit_transform(X_train_std)
            X_test_reduced = kpca.transform(X_test_std)

        # Create an SVM classifier
        svm_classifier = SVC()
        # Create GridSearchCV with the SVM classifier and hyperparameter grid
        grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, scoring='accuracy',n_jobs=-1)
        # Fit the grid search to the dimensionality-reduced training data
        grid_search.fit(X_train_reduced, y_train_std)
        # Get the best hyperparameters
        best_params = grid_search.best_params_
        # Train the final SVM model with the best hyperparameters on the full training data
        final_svm_model_std = SVC(**best_params)
        final_svm_model_std.fit(X_train_reduced, y_train_std)

        # Evaluate the SVM model's performance on the transformed testing data
        test_accuracy = accuracy_score(y_test_std, final_svm_model_std.predict(X_test_reduced))
        train_accuracy = accuracy_score(y_train_std, final_svm_model_std.predict(X_train_reduced))
        accuracy = np.mean([test_accuracy, train_accuracy])
        # Update the best combination if accuracy is higher
        if accuracy > best_std_accuracy:
            best_std_accuracy = accuracy
            best_std_technique = technique
            best_std_n_components = n_components

# Print the best combination and accuracy
print("std")
print("Best Dimensionality Reduction Technique:", best_std_technique)
print("Best n_components Value:", best_std_n_components)
print("Best Accuracy:", best_std_accuracy)

###########################################################################################################################
# NORM EXPERIMENT
########################################################################################################################### 
# Initialize variables to track the best combination
best_norm_technique = None
best_norm_n_components = None
best_norm_accuracy = 0.0  # Initialize with a low value

# Experiment with PCA, LDA, and KPCA
for n_components in n_components_values:
    for technique in ['pca', 'lda', 'kpca']:
        if technique == 'pca':
            pca = PCA(n_components=n_components)
            X_train_reduced = pca.fit_transform(X_train_norm)
            X_test_reduced = pca.transform(X_test_norm)
        elif technique == 'lda':
            lda = LDA(n_components=n_components, solver='eigen')
            X_train_reduced = lda.fit_transform(X_train_norm, y_train_norm)
            X_test_reduced = lda.transform(X_test_norm)
        elif technique == 'kpca':
            kpca = KernelPCA(kernel='rbf', n_components=n_components)
            X_train_reduced = kpca.fit_transform(X_train_norm)
            X_test_reduced = kpca.transform(X_test_norm)
        # Create an SVM classifier
        svm_classifier = SVC()
        # Create GridSearchCV with the SVM classifier and hyperparameter grid
        grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, scoring='accuracy',n_jobs=-1)
        # Fit the grid search to the dimensionality-reduced training data
        grid_search.fit(X_train_reduced, y_train_norm)
        # Get the best hyperparameters
        best_params = grid_search.best_params_
        # Train the final SVM model with the best hyperparameters on the full training data
        final_svm_model_norm = SVC(**best_params)
        final_svm_model_norm.fit(X_train_reduced, y_train_norm)
        # Evaluate the SVM model's performance on the transformed testing data
        test_accuracy = accuracy_score(y_test_norm, final_svm_model_norm.predict(X_test_reduced))
        train_accuracy = accuracy_score(y_train_norm, final_svm_model_norm.predict(X_train_reduced))
        accuracy = np.mean([test_accuracy, train_accuracy])
        # Update the best combination if accuracy is higher
        if accuracy > best_norm_accuracy:
            best_norm_accuracy = accuracy
            best_norm_technique = technique
            best_norm_n_components = n_components

# Print the best combination and accuracy
print("norm")
print("Best Dimensionality Reduction Technique:", best_norm_technique)
print("Best n_components Value:", best_norm_n_components)
print("Best Accuracy:", best_norm_accuracy)
###########################################################################################################################
# lets get the data reduced with the best technique
###########################################################################################################################
print(X_train_reduced)
print(X_train_norm)

if (best_std_accuracy > best_norm_accuracy):
    bestparams = final_svm_model_std.get_params()
    best_classifier = SVC(**bestparams)

    best_accuracy = best_std_accuracy
    best_technique = best_std_technique
    best_n_components = best_std_n_components
    X_train = X_train_std
    X_test = X_test_std
    y_train = y_train_std
    y_test = y_test_std
else:
    bestparams = final_svm_model_norm.get_params()
    best_classifier = SVC(**bestparams)
    best_accuracy = best_norm_accuracy
    best_technique = best_norm_technique
    best_n_components = best_norm_n_components
    X_train = X_train_norm
    X_test = X_test_norm
    y_train = y_train_norm
    y_test = y_test_norm

txt_text = ''
print(X_train.shape)
if (best_technique == 'pca'):
    pca = PCA(n_components=best_n_components)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    txt_text = "Most Important Features for PCA:\n" + str(pca.components_) + "\n\n"
elif (best_technique == 'lda'):
    lda = LDA(n_components=best_n_components, solver='eigen')
    X_train_reduced = lda.fit_transform(X_train, y_train)
    X_test_reduced = lda.transform(X_test)
    txt_text = "Most Important Features for LDA:\n" + str(lda.explained_variance_ratio_) + "\n\n"
elif (best_technique == 'kpca'):
    kpca = KernelPCA(kernel='rbf', n_components=best_n_components)
    X_train_reduced = kpca.fit_transform(X_train)
    X_test_reduced = kpca.transform(X_test)
    txt_text = "Most Important Features for PCA:\n" + str(kpca.eigenvalues_) + "\n\n"
print(X_train_reduced.shape)

###########################################################################################################################
# text
###########################################################################################################################

# Open the text file for writing
with open(TEXT_FILE_LOCATION, "w") as file:
    # Print most important features for PCA
    file.write(txt_text)
    # Print training and testing accuracies
    file.write("the best acc (SVM): {:.2f}\n".format(best_accuracy))

print(f"Results saved to '{TEXT_FILE_LOCATION}'.")

###########################################################################################################################
# plot
###########################################################################################################################
# Assuming you have model_lda correctly trained on LDA-transformed data
X_combined =np.vstack((X_test))
X_combined = X_combined.astype(float)
best_classifier.fit(X_train_reduced[:, :2], y_train)
print(X_test.shape)
print(y_test.shape)
print(X_train_reduced.shape)
plot_decision_regions(X=X_test_reduced[:, :2], y=y_test, clf=best_classifier)

# Plot with labels and legend
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')
plt.savefig(PLOT_LOCATION)