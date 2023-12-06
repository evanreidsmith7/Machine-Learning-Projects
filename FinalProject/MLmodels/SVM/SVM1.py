# import requirements
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
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
   plt.title('ROC AUC')
   plt.legend(loc="lower right")
   plt.savefig(save_location)
   #plt.show()
   plt.close()

##########################################################################################################################
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

################################################################################################################################################
# Data preprocessing
################################################################################################################################################

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
# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1, stratify=y)

from joblib import load

print("Loading model...")
# Load the saved model from a file
best_pipeline = load(model_save_location)

# Now you can use the loaded_model for predictions


print("model loaded")
# Predict on test data
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.decision_function(X_test)  # Calculate decision function values for ROC curve
y_pred_train = best_pipeline.predict(X_train)

# Evaluate the model
print("Model predictions done \n gathering metrics from model now...")

plot_confusion_matrix(y_test, y_pred, test_confusion_matrix_save_location, 'Test Confusion Matrix')
write_metrics(y_test, y_pred, y_train, y_pred_train, txt_location)
print("Metrics gathered")
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

