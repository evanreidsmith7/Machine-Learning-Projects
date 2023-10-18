#************************************************************************************
# Evan Smith
# ML – HW#4, Part 2
# Filename: hw4-part2.py
# Due: , 2023
#
# Objective:
#Use k-means++ to observe clusters in the data using the LEAP cluster
#• Combine all the mini datasets into a single dataset
#• Determine all the same requirements from Part 1
#*************************************************************************************
# Import libraries
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from kneed import KneeLocator
##########################################################################################################################
# functions
###########################################################################################################################

#create a list of colors for scatter plot
colors = ['green', 'blue', 'yellow', 'orange', 'black', 'purple', 'pink', 'brown', 'gray', 'cyan']

#create a list of n_clusters for KMeans
clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

def plot_clusters(X, n_clusters, file_path: str, title: str = 'Clusters'):
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
   plt.savefig(file_path)
   plt.close()

#########################################################################################################################

def plot_silhouette(X, n_clusters, file_path: str, title: str = 'Silhouette'):
   plt.xlim([-0.1, 1])
      # The (n_clusters+1)*10 is for inserting blank space between silhouette
      # plots of individual clusters, to demarcate them clearly.
   plt.ylim([0, len(X) + (n_clusters + 1) * 10])
   clusterer = KMeans(n_clusters=n_clusters,
                  init='k-means++',
                  n_init=10,
                  max_iter=300,
                  tol=1e-04,
                  random_state=0)
   cluster_labels = clusterer.fit_predict(X)
   silhouette_avg = silhouette_score(X, cluster_labels)
   print(
      "For n_clusters =",
      n_clusters,
      "The average silhouette_score is :",
      silhouette_avg,
   )

   # Compute the silhouette scores for each sample
   sample_silhouette_values = silhouette_samples(X, cluster_labels)

   y_lower = 10
   for i in range(n_clusters):
      # Aggregate the silhouette scores for samples belonging to
      # cluster i, and sort them
      ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

      ith_cluster_silhouette_values.sort()

      size_cluster_i = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_i

      color = cm.nipy_spectral(float(i) / n_clusters)
      plt.fill_betweenx(
         np.arange(y_lower, y_upper),
         0,
         ith_cluster_silhouette_values,
         facecolor=color,
         edgecolor=color,
         alpha=0.7,
      )

      # Label the silhouette plots with their cluster numbers at the middle
      plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

      # Compute the new y_lower for next plot
      y_lower = y_upper + 10  # 10 for the 0 samples

   plt.title(title)
   plt.xlabel("The silhouette coefficient values")
   plt.ylabel("Cluster label")

   # The vertical line for average silhouette score of all the values
   plt.axvline(x=silhouette_avg, color="red", linestyle="--")

   plt.yticks([])  # Clear the yaxis labels / ticks
   plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
   plt.savefig(file_path)
   plt.close()

#########################################################################################################################

def calculate_distortions(data, max_clusters):
    distortions = []
    for i in range(1, max_clusters+1):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(data)
        distortions.append(km.inertia_)
    return distortions

#########################################################################################################################

def plot_elbow_method(distortions, title: str, file_path: str):
   plt.plot(range(1, len(distortions)+1), distortions, marker='o')
   plt.title(title)
   plt.xlabel('Number of clusters')
   plt.ylabel('Distortion')
   plt.savefig(file_path)
   plt.close()

##########################################################################################################################

def get_distortion_score(data, n_clusters):
   km = KMeans(n_clusters=n_clusters,
               init='k-means++',
               n_init=10,
               max_iter=300,
               tol=1e-04,
               random_state=0)
   km.fit(data)
   score = km.inertia_
   return score

##########################################################################################################################

def write_distortion_scores(score1, file_path):
   print(file_path + '\n')
   print('Distortion Score: ' + str(score1) + '\n')
   with open(file_path, 'w') as file:
      file.write('Distortion Score: ' + str(score1) + '\n')

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
elbow_file_path_all = "Part2/Results/Elbows/elbow.png"

clusters_file_path_all = "Part2/Results/Clusters/clusters.png"

silhouettes_file_path_all = "Part2/Results/Silhouettes/silhoute.png"

distortion_score_file_path = "Part2/Results/DistortionScores/distortion_scores.txt"

# Use the pandas.read_csv() function to read the CSV file into a DataFrame
df_2011 = pd.read_csv(csv_file_path_2011, header=None)
df_2012 = pd.read_csv(csv_file_path_2012, header=None)
df_2013 = pd.read_csv(csv_file_path_2013, header=None)
df_2014 = pd.read_csv(csv_file_path_2014, header=None)
df_2015 = pd.read_csv(csv_file_path_2015, header=None)

df_2011[1:] = df_2011[1:].astype(float)
df_2012[1:] = df_2012[1:].astype(float)
df_2013[1:] = df_2013[1:].astype(float)
df_2014[1:] = df_2014[1:].astype(float)
df_2015[1:] = df_2015[1:].astype(float)

X_2011 = np.array(df_2011[1:])
X_2012 = np.array(df_2012[1:])
X_2013 = np.array(df_2013[1:])
X_2014 = np.array(df_2014[1:])
X_2015 = np.array(df_2015[1:])


# combine all of the data into one array
X_all = np.concatenate((X_2011, X_2012, X_2013, X_2014, X_2015), axis=0)


# standardize the data
std = StandardScaler()
X_all_std = std.fit_transform(X_all)

# PCA
pca = PCA(n_components=2)
X_all_pca = pca.fit_transform(X_all_std)


##########################################################################################################################
print("Data Preprocessing done.")
# Determine the number of centroids by using the Elbow Method
print("Determining the number of centroids by using the Elbow Method...")
###########################################################################################################################

# calculate distortion for a range of number of cluster
max_clusters = 10
distortions_all = calculate_distortions(X_all_pca, max_clusters)

##########################################################################################################################
print("done.")
# plot the elbows
print("plotting elbows...")
##########################################################################################################################

plot_elbow_method(distortions_all, 'Elbow Method for all', elbow_file_path_all)
k_elbow = KneeLocator(x=range(1, 11), y=distortions_all, curve='convex', direction='decreasing')
print("The optimal number of clusters is", k_elbow.elbow)
n_clusters_all = k_elbow.elbow

##########################################################################################################################
print("done.")
# plot the clusters
print("plotting clusters...")
##########################################################################################################################

plot_clusters(X_all_pca, n_clusters_all, silhouettes_file_path_all, 'Clusters for all')

##########################################################################################################################
print("done.")
# plot the silhouettes
print("plotting silhouettes...")
##########################################################################################################################

plot_silhouette(X_all_pca, n_clusters_all, silhouettes_file_path_all, 'Silhouette for all')

##########################################################################################################################
print("done.")
# get the distortion scores
print("calculating distortion scores...")
##########################################################################################################################

score_all = get_distortion_score(X_all_pca, n_clusters_all)

##########################################################################################################################
print("done.")
# get the distortion scores
print("writing distortion scores...")
##########################################################################################################################

write_distortion_scores(score_all, distortion_score_file_path)

##########################################################################################################################
print("\n\n\n\n\n\n\ndone.")
##########################################################################################################################