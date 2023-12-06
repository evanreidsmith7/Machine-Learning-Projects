# Import libraries
print("Importing libraries...")
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from joblib import dump
################################################################################################################################################
# Functions
################################################################################################################################################
##########################################################################################################################
#  ROC curve and AUC
# Create and save ROC AUC graph
def plot_roc_auc(y_test, y_prob, save_location):
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
   plt.title('SVM - ROC AUC')
   plt.legend(loc="lower right")
   plt.savefig(save_location)
   #plt.show()
   plt.close()

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

def save_model(model, save_location):
   print("Saving model...")
   dump(model, save_location)
   print("done...")
#*************************************************************************************
################################################################################################################################################
# Data preprocessing
################################################################################################################################################
print("Data preprocessing...")
train_confusion_matrix_save_location = 'MLmodels/SVM/Results/train_confusion_matrix.png'
test_confusion_matrix_save_location = 'MLmodels/SVM/Results/test_confusion_matrix.png'
txt_location = 'MLmodels/SVM/Results/metrics.txt'
loss_acc_plot_location = 'MLmodels/SVM/Results/acc_loss_plot.png'
roc_curve_save_location = 'MLmodels/SVM/Results/SVM_ROC.png'
model_save_location = 'MLmodels/SVM/Results/best_model.joblib'

region1_path = r"Datasets/EVANSDATA/R1"
region2_path = r"Datasets/EVANSDATA/R2"
region3_path = r"Datasets/EVANSDATA/R3"
region4_path = r"Datasets/EVANSDATA/R4"
region5_path = r"Datasets/EVANSDATA/R5"


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
# Best Parameters2: {'lda__n_components': 3, 'svm__C': 100, 'svm__gamma': 0.1, 'svm__kernel': 'rbf'}
# Define the pipeline


# Define the pipeline
lda_svm_pipeline = Pipeline([('scaler', StandardScaler()),
                          ('lda', LDA(n_components=3)),
                          ('svm', SVC(C=100, gamma=0.1, kernel='rbf'))])

# Define the parameter grid for GridSearch
#best params 646am 12/6/23
param_grid2 = {
    'svm__kernel': ['rbf'],
}

grid_search2 = GridSearchCV(
    estimator=lda_svm_pipeline,
    param_grid=param_grid2,
    cv=4,                              # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,                         # Use all available CPU cores
    verbose=1                          # Print detailed progress
)
grid_search2.fit(X_train_scaled, y_train)
best_pipeline = grid_search2.best_estimator_

print("Best Parameters2:", grid_search2.best_params_)

# Fit the best pipeline with the entire training data
best_pipeline.fit(X_train, y_train)

save_model(best_pipeline, model_save_location)

# Predict on test data
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.decision_function(X_test)  # Calculate decision function values for ROC curve
y_pred_train = best_pipeline.predict(X_train)

# Evaluate the model
print("Model predictions done \n gathering metrics from model now...")

plot_confusion_matrix(y_test, y_pred, test_confusion_matrix_save_location, 'Test Confusion Matrix')
write_metrics(y_test, y_pred, y_train, y_pred_train, txt_location)

# Plot ROC curve and AUC
plot_roc_auc(y_test, y_prob, roc_curve_save_location)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted') 
recall = recall_score(y_test, y_pred, average='weighted') 
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

