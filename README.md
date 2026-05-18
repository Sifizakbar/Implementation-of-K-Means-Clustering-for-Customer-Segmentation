# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries for data handling, clustering, and visualization.
2.Load the customer dataset and select the Annual Income and Spending Score features.
3.Create the K-Means clustering model with the required number of clusters.
4.Train the K-Means model and assign each data point to a cluster.
5.Store the predicted cluster labels for all customers.
6.Plot the customer data points with different colors based on their assigned clusters.
7.Display the cluster centroids and add labels and a title to visualize customer segmentation clearly.
## Program:
```

Developed by: Sifiz A
RegisterNumber:  212225040414

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

X = data.iloc[:, [3, 4]].values

kmeans = KMeans(n_clusters=5, random_state=0)

y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50)

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            marker='X')

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.show()

```

## Output:
![K Means Clustering for Customer Segmentation](sam.png)

<img width="1504" height="723" alt="Screenshot (466)" src="https://github.com/user-attachments/assets/345437de-dad3-43a6-8e7f-9a044ccf65e7" />
<img width="1498" height="667" alt="Screenshot (467)" src="https://github.com/user-attachments/assets/99284d34-28ad-4380-bcd1-e7673e2af98a" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
