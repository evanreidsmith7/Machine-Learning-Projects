# Import libraries
print("Importing libraries...")
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from kneed import KneeLocator
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import tensorflow as tf

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("No GPU devices available. Make sure TensorFlow is installed with GPU support.")
else:
    print("GPU is available.")
################################################################################################################################################
# Functions
################################################################################################################################################
#create a list of colors for scatter plot
colors = ['green', 'blue', 'yellow', 'orange', 'black', 'purple', 'pink', 'brown', 'gray', 'cyan']

#create a list of n_clusters for KMeans
clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

def plot_clusters(X, n_clusters, save_location: str, title: str = 'Clusters'):
    km = KMeans(n_clusters=n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)
    y_km = km.fit_predict(X)

    for i in range(n_clusters):
        plt.scatter(X[y_km == i, 0],
                    X[y_km == i, 1],
                    s=5,
                    c=colors[i],
                    marker='s',
                    edgecolor='black',
                    label='cluster ' + str(i + 1))
        
    plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=50, marker='*',
            c='red', edgecolor='black',
            label='centroids')
    plt.legend(scatterpoints=1)
    plt.title(title)
    plt.grid()
    plt.savefig(save_location)
    plt.close()

#*************************************************************************************
# Function to load CSV files from a directory
def load_csv_files(directory):
    dataset = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            dataset = pd.concat([dataset, df], ignore_index=True)
    return dataset

def plot_confusion_matrix(y_labels, y_preds, save_location, title):
   print("Plotting " + title + " confusion matrix...")
   confusion_data = confusion_matrix(y_labels, y_preds)
   plt.figure(figsize=(8, 6))
   sns.heatmap(confusion_data, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
   plt.xlabel("Predicted")
   plt.ylabel("True")
   plt.title(title)
   plt.savefig(save_location)
   #plt.show()
   plt.close()
   print("done...")

#*************************************************************************************

def plot_loss_acc(history, save_location):
   print("Plotting loss and accuracy...")
   acc = history.history['acc']
   val_acc = history.history['val_acc']
   loss = history.history['loss']
   val_loss = history.history['val_loss']
   epochs = range(1, len(acc) + 1)
   # Create subplots
   plt.figure(figsize=(12, 6))

   # Subplot for accuracy
   plt.subplot(1, 2, 1)
   plt.plot(epochs, acc, 'bo', label='Training acc')
   plt.plot(epochs, val_acc, 'b', label='Validation acc')
   plt.title('Training and validation accuracy')
   plt.legend()

   # Subplot for loss
   plt.subplot(1, 2, 2)
   plt.plot(epochs, loss, 'bo', label='Training loss')
   plt.plot(epochs, val_loss, 'b', label='Validation loss')
   plt.title('Training and validation loss')
   plt.legend()

   plt.savefig(save_location)
   plt.close()
   print("done...")

#*************************************************************************************

def write_metrics(y_test, y_test_pred, y_train, y_train_pred, txt_location):
   # Calculate and print the accuracy
   test_accuracy = accuracy_score(y_test, y_test_pred)
   print("test Accuracy: %.2f%%" % (test_accuracy * 100.0))
   train_accuracy = accuracy_score(y_train, y_train_pred)
   print("train Accuracy: %.2f%%" % (train_accuracy * 100.0))
   print("\n\n\n")
   classification_report_str = classification_report(y_test, y_test_pred, target_names=['1', '2', '3', '4', '5'])
   print(classification_report_str)
   with open(txt_location, 'w') as file:
      file.write("test Accuracy: %.2f%%" % (test_accuracy * 100.0))
      file.write("train Accuracy: %.2f%%" % (train_accuracy * 100.0))
      file.write("\n\n\n")
      file.write(classification_report_str)

#*************************************************************************************
################################################################################################################################################
# Data preprocessing
################################################################################################################################################
print("Data preprocessing...")
train_confusion_matrix_save_location = 'ANNmodels/Results/train_confusion_matrix.png'
test_confusion_matrix_save_location = 'ANNmodels/Results/test_confusion_matrix.png'
txt_location = 'ANNmodels/Results/metrics.txt'
loss_acc_plot_location = 'ANNmodels/Results/acc_loss_plot.png'
model_save_location = 'ANNmodels/Results/'

region1_path = r"Datasets/FinalDataset/R1"
region2_path = r"Datasets/FinalDataset/R2"
region3_path = r"Datasets/FinalDataset/R3"
region4_path = r"Datasets/FinalDataset/R4"
region5_path = r"Datasets/FinalDataset/R5"


region_paths = [
    region1_path,
    region2_path,
    region3_path,
    region4_path,
    region5_path,
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


y = full_dataset["Region"]
# encode y
le = LabelEncoder()
y_encoded = le.fit_transform(y)



X = full_dataset.drop(["Region"], axis=1)
# Convert empty strings to NaN
X[X == ''] = np.nan

# Convert all values to float
X = X.astype(float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("done...\n\n\n\n")

# Initialize PCA, here we are reducing the data to 2 dimensions for easy visualization
#FIGURE PCA OUT
pca = PCA(n_components=2)

# Fit and transform the data
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

##########################################################################################################################
# Create the model
##########################################################################################################################

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}

# Create an SVM model
svm_model = SVC(random_state=1)

# Create a GridSearchCV object
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train_pca, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best hyperparameters
best_model = SVC(**best_params, random_state=1)
best_model.fit(X_train_pca, y_train)

# Predict on test data
y_pred = best_model.predict(X_test_pca)
y_pred_train = best_model.predict(X_train_pca)

# Evaluate the model
print("Model predictions done \n gathering metrics from model now...")

plot_confusion_matrix(y_test, y_pred, test_confusion_matrix_save_location, 'Test Confusion Matrix')

write_metrics(y_test, y_pred, y_train, y_pred_train, txt_location)
