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
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
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
clusters_file_path_2012 = "Part1/Results/2012clusters.png"
clusters_file_path_2013 = "Part1/Results/2013clusters.png"
clusters_file_path_2014 = "Part1/Results/2014clusters.png"
clusters_file_path_2015 = "Part1/Results/2015clusters.png"

silhouettes_file_path_2011 = "Part1/Results/2011silhoute.png"
silhouettes_file_path_2011 = "Part1/Results/2012silhoute.png"
silhouettes_file_path_2011 = "Part1/Results/2013silhoute.png"
silhouettes_file_path_2011 = "Part1/Results/2014silhoute.png"
silhouettes_file_path_2011 = "Part1/Results/2015silhoute.png"


# Use the pandas.read_csv() function to read the CSV file into a DataFrame
df_2011 = pd.read_csv(csv_file_path_2011, header=None)
df_2012 = pd.read_csv(csv_file_path_2012, header=None)
df_2013 = pd.read_csv(csv_file_path_2013, header=None)
df_2014 = pd.read_csv(csv_file_path_2014, header=None)
df_2015 = pd.read_csv(csv_file_path_2015, header=None)
print(df_2011.head())


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
    km.fit(df_2011)
    distortions.append(km.inertia_)

##########################################################################################################################
print("done.")
# plot the distortion
print("plotting distortions...")
##########################################################################################################################

plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig(distortion_clusters_file_path)

##########################################################################################################################
print("done.")
# plot the clusters
print("plotting clusters...")
##########################################################################################################################