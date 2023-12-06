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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras import backend as K
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.regularizers import l2
from sklearn.manifold import TSNE
from umap import UMAP


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
def plot_tsne(X, y, save_location):
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
   plt.savefig(save_location)
#*************************************************************************************

def plot_umap(X, y, save_location):
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
   plt.savefig(save_location)

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
print("Data preprocessing...")
train_confusion_matrix_save_location = 'ANNmodels/EvanResults/annII/train_confusion_matrix.png'
test_confusion_matrix_save_location = 'ANNmodels/EvanResults/annII/test_confusion_matrix.png'
txt_location = 'ANNmodels/EvanResults/annII/metrics.txt'
loss_acc_plot_location = 'ANNmodels/EvanResults/annII/acc_loss_plot.png'
model_save_location = 'ANNmodels/EvanResults/'
roc_curve_save_location = 'ANNmodels/EvanResults/annII/roc_curve.png'
tsne_save_location = 'ANNmodels/EvanResults/annII/tsne.png'
umap_save_location = 'ANNmodels/EvanResults/annII/umap.png'

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

print("X: ", X.shape)
print("y: ", y_encoded.shape)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("prepocessing done.../n/n/n/n")
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
# Create the model
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

















print("Creating the model...")

model = Sequential()

# Input layer 762 neurons (number of features)
model.add(Dense(units=762, input_dim=762, activation='relu'))

# Hidden layers
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))  # Adjust dropout rate as needed
model.add(Dense(units=16, activation='relu'))
# Output layer
model.add(Dense(units=5, activation='softmax'))

print("done...\n\n\n\n")
##########################################################################################################################
# Compile the model
##########################################################################################################################
print("Compiling the model...")

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

print("done...\n\n\n\n")
##########################################################################################################################
# Train the model
##########################################################################################################################
print("Training the model...")

history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data =(X_test_scaled, y_test), verbose=2)

print("done...\n\n\n\n")


























##########################################################################################################################
# Evaluate the model
##########################################################################################################################

print("Evaluating the model...")

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Test Accuracy: {accuracy*100:.2f}%")

print("done...\n\n\n\n")

# Saving the model for Future Inferences
model_json = model.to_json()
with open(model_save_location + "model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_save_location + "model.h5")


y_test_pred = model.predict(X_test_scaled)
print("y_test_pred:", y_test_pred)
print("\n\n\ny:", y_test)


y_test_pred = np.array(y_test_pred)
predicted_classes = np.argmax(y_test_pred, axis=1)
print("predicted_classes:", predicted_classes)

y_train_pred = model.predict(X_train_scaled)
y_train_pred = np.array(y_train_pred)

predicted_classes_train = np.argmax(y_train_pred, axis=1)
predicted_classes = np.argmax(y_test_pred, axis=1)
print("predicted_classes:", predicted_classes)

#*************************************************************************************
# Evaluate model
#*************************************************************************************
print("Model predictions done \n gathering metrics from model now...")

plot_confusion_matrix(y_test, predicted_classes, test_confusion_matrix_save_location, 'Test Confusion Matrix')

plot_loss_acc(history, loss_acc_plot_location)

write_metrics(y_test, predicted_classes, y_train, predicted_classes_train, txt_location)

plot_roc_auc(y_test, y_test_pred, roc_curve_save_location)