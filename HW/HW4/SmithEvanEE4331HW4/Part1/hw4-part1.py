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

##########################################################################################################################
# Data Preprocessing
print("Data Preprocessing...")
###########################################################################################################################
# Specify the path to your CSV file
csv_file_path = "Dataset/gt_2011.csv"

# Use the pandas.read_csv() function to read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
print(df.head())