#************************************************************************************
# Evan Smith
# ML – HW#4, Part 1
# Filename: hw4-part1.py
# Due: , 2023
#
# Objective:
#• Use k-means++ to observe clusters in the data using the LEAP cluster
#• Determine the number of centroids by using the Elbow Method (provide the plot) for the
#  2011 dataset
#• Use the correct number of centroids and plot the clusters with its centers and silhouettes
#  for each individual year
#  Determine the distortion score and save it to a text file for each individual year
#*************************************************************************************
# Import libraries
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import numpy as np
##########################################################################################################################
# Data Preprocessing
print("Data Preprocessing...")
###########################################################################################################################
# Specify the path to your CSV file
csv_file_path_2011 = "Dataset/gt_2011.csv"
csv_file_path_2012 = "Dataset/gt_2012.csv"
csv_file_path_2013 = "Dataset/gt_2013.csv"
csv_file_path_2014 = "Dataset/gt_2014.csv"
csv_file_path_2015 = "Dataset/gt_2015.csv"

# image paths
distortion_clusters_file_path = "Part1/Results/distortion_clusters.png"

clusters_file_path_2011 = "Part1/Results/2011clusters.png"
clusters_file_path_2011a = "Part1/Results/2011clusters_a.png"
clusters_file_path_2012 = "Part1/Results/2012clusters.png"
clusters_file_path_2013 = "Part1/Results/2013clusters.png"
clusters_file_path_2014 = "Part1/Results/2014clusters.png"
clusters_file_path_2015 = "Part1/Results/2015clusters.png"

silhouettes_file_path_2011 = "Part1/Results/2011silhoute.png"
silhouettes_file_path_2012 = "Part1/Results/2012silhoute.png"
silhouettes_file_path_2013 = "Part1/Results/2013silhoute.png"
silhouettes_file_path_2014 = "Part1/Results/2014silhoute.png"
silhouettes_file_path_2015 = "Part1/Results/2015silhoute.png"


# Use the pandas.read_csv() function to read the CSV file into a DataFrame
df_2011 = pd.read_csv(csv_file_path_2011, header=None)
df_2012 = pd.read_csv(csv_file_path_2012, header=None)
df_2013 = pd.read_csv(csv_file_path_2013, header=None)
df_2014 = pd.read_csv(csv_file_path_2014, header=None)
df_2015 = pd.read_csv(csv_file_path_2015, header=None)
print(df_2011.head())
df_2011[1:] = df_2011[1:].astype(float)

X_2011 = np.array(df_2011[1:])   

##########################################################################################################################
print("Data Preprocessing done.")
# Determine the number of centroids by using the Elbow Method
print("Determining the number of centroids by using the Elbow Method...")
###########################################################################################################################

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(df_2011[1:])
    print("\n"+ str(distortions)+"\n")
    distortions.append(km.inertia_)

print(distortions)
##########################################################################################################################
print("done.")
# plot the distortion
print("plotting distortions...")
##########################################################################################################################

plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig(distortion_clusters_file_path)
plt.close()


##########################################################################################################################
print("done.")
# plot the clusters
print("plotting clusters...")
##########################################################################################################################
# use pca for 2d plot
sc = StandardScaler()
X_std = sc.fit_transform(X_2011)

pca = PCA(n_components=2)
X_pca_2011 = pca.fit_transform(X_std)
km = KMeans(n_clusters=3,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42)

y_km = km.fit_predict(X_pca_2011)
print("\n\n" + str(km.inertia_)) 

plt.scatter(X_pca_2011[y_km == 0, 0],
            X_pca_2011[y_km == 0, 1],
            s=10, c='lightgreen',
            marker='s', edgecolor='black',
            label='cluster 1')
plt.scatter(X_pca_2011[y_km == 1, 0],
            X_pca_2011[y_km == 1, 1],
            s=10, c='orange',
            marker='o', edgecolor='black',
            label='cluster 2')
plt.scatter(X_pca_2011[y_km == 2, 0],
            X_pca_2011[y_km == 2, 1],
            s=10, c='lightblue',
            marker='v', edgecolor='black',
            label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=25, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.savefig(clusters_file_path_2011)
plt.close()






##########################################################################################################################
print("done.")
# plot the clusters
print("plotting clusters...")
##########################################################################################################################

km = KMeans(n_clusters=5,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42)

y_km = km.fit_predict(X_pca_2011)
print("\n\n" + str(km.inertia_)) 

plt.scatter(X_pca_2011[y_km == 0, 0],
            X_pca_2011[y_km == 0, 1],
            s=10, c='lightgreen',
            marker='s', edgecolor='black',
            label='cluster 1')
plt.scatter(X_pca_2011[y_km == 1, 0],
            X_pca_2011[y_km == 1, 1],
            s=10, c='orange',
            marker='o', edgecolor='black',
            label='cluster 2')
plt.scatter(X_pca_2011[y_km == 2, 0],
            X_pca_2011[y_km == 2, 1],
            s=10, c='lightblue',
            marker='v', edgecolor='black',
            label='cluster 3')
plt.scatter(X_pca_2011[y_km == 3, 0],
            X_pca_2011[y_km == 3, 1],
            s=10, c='orange',
            marker='o', edgecolor='black',
            label='cluster 4')
plt.scatter(X_pca_2011[y_km == 4, 0],
            X_pca_2011[y_km == 4, 1],
            s=10, c='red',
            marker='v', edgecolor='black',
            label='cluster 5')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=25, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.savefig(clusters_file_path_2011a)
plt.close()
##########################################################################################################################
print("done.")
# plot the silhouettes
print("plotting silhouettes...")
##########################################################################################################################