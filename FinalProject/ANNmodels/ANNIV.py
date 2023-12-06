from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Function to load CSV files from a directory
def load_csv_files(directory):
    dataset = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            dataset = pd.concat([dataset, df], ignore_index=True)
    return dataset

# Function to plot ROC curve and save the graph
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
    plt.close()

# Function to plot confusion matrix and save the graph
def plot_confusion_matrix(y_labels, y_preds, save_location, title):
    print("Plotting " + title + " confusion matrix...")
    confusion_data = confusion_matrix(y_labels, y_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_data, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.savefig(save_location)
    plt.close()
    print("done...")

# Function to plot loss and accuracy and save the graph
def plot_loss_acc(history, save_location):
    print("Plotting loss and accuracy...")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
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

# Function to write metrics to a text file
def write_metrics(y_test, y_test_pred, y_train, y_train_pred, txt_location):
    # Calculate and print the accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train Accuracy: %.2f%%" % (train_accuracy * 100.0))
    print("\n\n\n")
    classification_report_str = classification_report(y_test, y_test_pred, target_names=['1', '2', '3', '4', '5'])
    print(classification_report_str)
    with open(txt_location, 'w') as file:
        file.write("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
        file.write("Train Accuracy: %.2f%%" % (train_accuracy * 100.0))
        file.write("\n\n\n")
        file.write(classification_report_str)

# Data preprocessing
print("Data preprocessing...")
train_confusion_matrix_save_location = 'ANNmodels/Results/train_confusion_matrix.png'
test_confusion_matrix_save_location = 'ANNmodels/Results/test_confusion_matrix.png'
txt_location = 'ANNmodels/Results/metrics.txt'
loss_acc_plot_location = 'ANNmodels/Results/acc_loss_plot.png'
model_save_location = 'ANNmodels/Results/'
roc_curve_save_location = 'ANNmodels/Results/roc_curve.png'

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

print("done.../n/n/n/n")

##########################################################################################################################
# Create the model
##########################################################################################################################

# Create the model
print("Creating the model...")
model = Sequential()
model.add(Dense(32, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

# Compile the model
print("Compiling the model...")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
print("Training the model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=128,
                    validation_data=(X_test_scaled, y_test), verbose=2,
                    callbacks=[early_stopping])

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Saving the model for Future Inferences
model_json = model.to_json()
with open(model_save_location + "model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_save_location + "model.h5")

# Predictions
y_test_pred = model.predict(X_test_scaled)
predicted_classes = np.argmax(y_test_pred, axis=1)

y_train_pred = model.predict(X_train_scaled)
predicted_classes_train = np.argmax(y_train_pred, axis=1)

# Evaluate model
print("Model predictions done \n gathering metrics from model now...")

plot_confusion_matrix(y_test, predicted_classes, test_confusion_matrix_save_location, 'Test Confusion Matrix')
plot_loss_acc(history, loss_acc_plot_location)
write_metrics(y_test, predicted_classes, y_train, predicted_classes_train, txt_location)
plot_roc_auc(y_test, y_test_pred, roc_curve_save_location)
