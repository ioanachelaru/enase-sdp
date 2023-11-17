from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utilitaries import read_csv_file
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer

   

BUGS = read_csv_file('data_to_cluster/bugs_to_cluster.csv')
BUGS_IDs = [key for key in BUGS.keys()]
BUGS_VALUES = [[float(item) for item in value] for value in BUGS.values()]
X = np.array(BUGS_VALUES)


'''
Elbow method to find the optimal number of clusters
Tests the number of clusters from 2 to 10
Saves a plot of the elbow method in the clustering_images folder
'''
def elbow_clustering():
    km = KMeans(random_state=42)
    visualizer = KElbowVisualizer(km, k=(2,10))
    
    visualizer.fit(X)
    visualizer.show(outpath="clustering_images/elbow_plot_fancy.png")


'''
Silhouette method to find the optimal number of clusters
Used to validate the results of the elbow method
Tests the number of clusters from 2 to 10
Saves a plot of the silhouette method in the clustering_images folder
'''
def silhouette_clustering():
    fig, ax = plt.subplots(3, 2, figsize=(30,20))
    for i in range(2, 8):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        q, mod = divmod(i, 2)
        
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(X)
        visualizer.show(outpath="clustering_images/silhouette_plot_2-7.png") 


'''
Clusters the data from the CSV file
'''
def cluster_data():
    # Read the CSV file into a dictionary
    data_dict = read_csv_file('data_to_cluster/bugs_to_cluster.csv')

    # Convert the dictionary values to a 2D list
    data = [value for value in data_dict.values()]

    # Convert the 2D list to a numpy array
    X = np.array(data)

    # Create a KMeans object with 3 clusters
    kmeans = KMeans(n_clusters=3)

    # Fit the data to the `kmeans` object
    kmeans.fit(X)

    # Print the cluster centers
    print(kmeans.cluster_centers_)

    # Print the cluster labels
    print(kmeans.labels_)

    # Plotting the cluster centers and the data points on a 2D plane
    plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
    plt.savefig('cluster_plot.png')

if __name__ == '__main__':
    # elbow_clustering()
    # silhouette_clustering()
    cluster_data()