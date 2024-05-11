#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:53:19 2024

@author: 
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#***** Cluster Analysis Problem *****
#read data
data = pd.read_csv(r'/Users/sadafjawad/Desktop/CPS844/Assignment2/6_class_csv.csv')
#keep numerical data only for clustering
numericalData = data.drop(['Star color', 'Spectral Class', 'Radius(R/Ro)', 'Absolute magnitude(Mv)'], axis=1)
#transform y-data to reflect realistic H-R diagram visually
numericalData['Luminosity(L/Lo)'] = np.log10(numericalData['Luminosity(L/Lo)'])
#plot the H-R diagram with Temperature and Luminosity for visual purposes
plt.figure(figsize=(10, 8))
plt.scatter(numericalData['Temperature (K)'], numericalData['Luminosity(L/Lo)'], alpha=0.7, c='blue')
plt.xlabel('Temperature (K)')
plt.ylabel('Log Luminosity(L/Lo)')
plt.title('H-R Diagram of Dataset (Temperature vs Luminosity)')
plt.gca().invert_xaxis()
plt.show()

#scale x-data (temperature)
scaler = StandardScaler()
numericalData['Scaled Temperature'] = scaler.fit_transform(numericalData[['Temperature (K)']])
#keep the log-transformed 'Luminosity(L/Lo)' as it is
numericalData['Log Luminosity(L/Lo)'] = numericalData['Luminosity(L/Lo)']
#prepare the array for clustering
X_for_clustering = numericalData[['Scaled Temperature', 'Log Luminosity(L/Lo)']].values

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X_for_clustering)

#plot the H-R diagram with Temperature and Luminosity
plt.figure(figsize=(10, 8))
plt.scatter(numericalData['Temperature (K)'], numericalData['Luminosity(L/Lo)'], c=clusters, cmap='jet', alpha=0.7)
plt.xlabel('Temperature (K)')
plt.ylabel('Log Luminosity(L/Lo)')
plt.title('H-R Diagram of Dataset (Temperature vs Luminosity) with KMeans Clustering')
plt.scatter([], [], color='lightgreen', label='Supergiants/Giants')
plt.scatter([], [], color='brown', label='Main Sequence')
plt.scatter([], [], color='darkblue', label='Dwarfs')
plt.legend()
plt.gca().invert_xaxis()
plt.show()


