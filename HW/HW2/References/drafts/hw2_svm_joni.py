# Importing all required libraries
from sklearn import datasets
import numpy as np
import pandas as pd  # Add pandas library for reading CSV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Correct import statement
from sklearn.linear_model import Perceptron
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions

# Set the path of each of the folders we want to excract from
Corridor_rm155_71_loc0000_path = "Corridor_rm155_7.1/Loc_0000"
Lab139_71_loc0000_path = "Lab139_7.1/Loc_0000"
Main_Lobby71_loc0000_path = "Main_Lobby_7.1/Loc_0000"
Sport_Hall_71_loc0000_path = "Sport_Hall_7.1/Loc_0000"

def extract_label(file_name):
    if 'Corridor_rm155_71_loc0000_path' in file_name:
        return 'Corridor'
    elif 'Lab139_71_loc0000_path' in file_name:
        return 'Lab'
    elif 'Main_Lobby71_loc0000_path' in file_name:
        return 'Main Lobby'
    elif 'Sport_Hall_71_loc0000_path' in file_name:
        return "Sport Hall"
    else:
        return None
    
combined_data_sport = pd.DataFrame()

for filename in os.listdir(Sport_Hall_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Sport_Hall_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        # Split the values in the existing column by semicolons into new columns
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data_sport = pd.concat([combined_data_sport, data], ignore_index=True)

combined_data_sport.to_csv("combined_data_corridor.csv", index=False)

combined_data_corridor = pd.DataFrame()

for filename in os.listdir(Corridor_rm155_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Corridor_rm155_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        # Split the values in the existing column by semicolons into new columns
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data_corridor = pd.concat([combined_data_corridor, data], ignore_index=True)

combined_data_corridor.to_csv("combined_data_corridor.csv", index=False)

combined_data_lab1 = pd.DataFrame()

for filename in os.listdir(Lab139_71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Lab139_71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data_lab1 = pd.concat([combined_data_lab1, data], ignore_index=True)

combined_data_lab1.to_csv("data_lab.csv", index=False)

combined_data_main_lobby = pd.DataFrame()

for filename in os.listdir(Main_Lobby71_loc0000_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(Main_Lobby71_loc0000_path, filename)
        data = pd.read_csv(file_path)
        data = data['# Version 1.00'].str.split(';', expand=True)
        data = data.drop([0,1])
        combined_data_main_lobby = pd.concat([combined_data_main_lobby, data], ignore_index=True)

combined_data_main_lobby.to_csv("combined_data_corridor.csv", index=False)

combined_data_sport['label'] = "Sport"
combined_data_corridor['label'] = "Corridor"
combined_data_main_lobby['label'] = "Lobby"
combined_data_lab1['label'] = "lab"

data_frames = [combined_data_sport, combined_data_corridor, combined_data_main_lobby, combined_data_lab1]

combined_data = pd.concat(data_frames, ignore_index=True)

combined_data.to_csv("combined_data_all.csv", index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Handling missing values by dropping rows with missing values
combined_data.dropna(inplace=True)


# Encode labels using LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
combined_data['label_encoded'] = label_encoder.fit_transform(combined_data['label'])

# Split the data into features (X) and labels (y_encoded)
X = combined_data.drop(columns=['label', 'label_encoded'])
y_encoded = combined_data['label_encoded']


# Convert empty strings to NaN
X[X == ''] = np.nan

# Convert the entire array to float
X = X.astype(float)

X = X.iloc[:, :-1]  # Remove the last column

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
n_components_pca = 4  # Adjust the number of components as needed
pca = PCA(n_components=n_components_pca)
X_pca = pca.fit_transform(X_scaled)

# Apply LDA
n_components_lda = 2  # Adjust the number of components as needed
lda = LDA(n_components=n_components_lda)
X_lda = lda.fit_transform(X_scaled, y_encoded)

# Split the transformed data into training and testing sets
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)
X_train_lda, X_test_lda, y_train, y_test = train_test_split(X_lda, y_encoded, test_size=0.2, random_state=42)

# Train SVM models using PCA and LDA-reduced data
model_pca = SVC(kernel='linear', C=1.0, random_state=42)
model_pca.fit(X_train_pca, y_train)

model_lda = SVC(kernel='linear', C=1.0, random_state=42)
model_lda.fit(X_train_lda, y_train)

# Make predictions and calculate accuracy for both SVM models
y_pred_pca = model_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

y_pred_lda = model_lda.predict(X_test_lda)
accuracy_lda = accuracy_score(y_test, y_pred_lda)

print(f"Accuracy with PCA (SVM): {accuracy_pca:.2f}")
print(f"Accuracy with LDA (SVM): {accuracy_lda:.2f}")

#t-SNE
########################################################################################
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# import umap
# Perform t-SNE
# Initialize t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

# Create a scatter plot
plt.figure(figsize=(8, 6))
# Scatter plot for each class label
for label in np.unique(y_encoded):
    plt.scatter(X_2d[y_encoded == label, 0], X_2d[y_encoded == label, 1], label=label)
plt.legend()
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('2D t-SNE')
# Show the plot
plt.savefig('tsne_plot.png')  
plt.show()

#UMAP
#####################################################################################
import umap

# Initialize UMAP
umap_model = umap.UMAP(n_components=2)
X_umap = umap_model.fit_transform(X)

# Create a scatter plot
plt.figure(figsize=(8, 6))
# Scatter plot for each class label (replace 'y_encoded' with your encoded labels)
for label in np.unique(y_encoded):
    plt.scatter(X_umap[y_encoded == label, 0], X_umap[y_encoded == label, 1], label=label)

plt.legend()
plt.xlabel('UMAP feature 1')
plt.ylabel('UMAP feature 2')
plt.title('2D UMAP Visualization')
# Save the UMAP plot as an image file
plt.savefig('umap_plot.png')
plt.show()

#TEXT FILE
#####################################################################################
# Define the filename for the output text file
output_filename = "feature_selection_and_accuracy.txt"

# Open the text file for writing
with open(output_filename, "w") as file:
    # Print most important features for PCA
    file.write("Most Important Features for PCA:\n")
    file.write(str(pca.components_) + "\n\n")

    # Print most important features for LDA
    file.write("Most Important Features for LDA:\n")
    file.write(str(lda.explained_variance_ratio_) + "\n\n")

    # Print training and testing accuracies
    file.write("Accuracy with PCA (SVM): {:.2f}\n".format(accuracy_pca))
    file.write("Accuracy with LDA (SVM): {:.2f}\n".format(accuracy_lda))

print(f"Results saved to '{output_filename}'.")

#DECISION PLOT

#####################################################################################
# Assuming you have model_lda correctly trained on LDA-transformed data
plot_decision_regions(X, y_encoded, clf=model_lda, legend=2)

# Plot with labels and legend
plt.xlabel('LDA Feature 1')
plt.ylabel('LDA Feature 2')
plt.legend(loc='upper left')
plt.show()

# Define the directory path for saving results
directory = 'decision_results'

# Check if the directory exists, and if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

# Save the plot under the 'Results' directory
plot_path = os.path.join(directory, 'decision_classification.png')
plt.savefig(plot_path)
