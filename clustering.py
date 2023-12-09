from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utilitaries import read_csv_file, write_list_to_file, write_dict_to_file
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.decomposition import PCA
   

BUGS = read_csv_file('parsed_data/bugs_to_cluster.csv')
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
    visualizer.show(outpath="clustering_images/elbow_plot.png")


'''
Silhouette method to find the optimal number of clusters
Used to validate the results of the elbow method
Tests the number of clusters from 2 to 10
Saves a plot of the silhouette method in the clustering_images folder


Average silhouette score for 2 centers:  0.17962307224129012
Average silhouette score for 3 centers:  0.17697695388105275
Average silhouette score for 4 centers:  0.1132155255619642
Average silhouette score for 5 centers:  0.10921873544136806
Average silhouette score for 6 centers:  0.10731665995852611
Average silhouette score for 7 centers:  0.0815382219412979

'''
def silhouette_clustering():
    fig, ax = plt.subplots(3, 2, figsize=(30,20))
    for i in range(2, 8):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        q, mod = divmod(i, 2)
        
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(X)
        print(f'Average silhouette score for {i} centers: ', visualizer.silhouette_score_)
        visualizer.show(outpath="clustering_images/silhouette_plot_2-7.png") 


'''
Labels the data for binary classification
The data is labeled with 1 if the bug is in the target cluster and 0 otherwise
'''
def label_data_for_binary_classification(target_cluster, target_file):
    all_data = read_csv_file('parsed_data/all_to_cluster.csv')
    clustered_bugs = read_csv_file(f'clustered_data/{target_file}')

    labeled_bugs = {key: ([1 if int(values[0]) == target_cluster else 0] + values[1:]) for key, values in clustered_bugs.items()}
    labeled_data = {key: [0] + values for key, values in all_data.items()}

    labeled_data.update(labeled_bugs)
    
    return labeled_data

'''
Clusters the data from the CSV file
Creates plots for the clusters
Saves the labeled data for each cluster in a CSV file
'''
def cluster_data(no_clusters=5):
    kmeans = KMeans(n_clusters=no_clusters)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    labeled_data = [[BUGS_IDs[i], y_kmeans[i]] + sublist for i, sublist in enumerate(BUGS_VALUES)]
    write_list_to_file(f'clustered_data/clustered_data_{no_clusters}_centers.csv', labeled_data)


    for i in range(0, no_clusters):
        data = label_data_for_binary_classification(i, f'clustered_data_{no_clusters}_centers.csv')
        write_dict_to_file(f'data_to_cluster/{no_clusters}_clusters/labeled_data_for_cluster_{i}.csv', data)


    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)

    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y_kmeans, s=50, cmap='viridis')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Clustering with {no_clusters} centers')

    plt.savefig(f'clustering_images/clustering_{no_clusters}_centers.png')
        

if __name__ == '__main__':
    # elbow_clustering()
    # silhouette_clustering()
    cluster_data(4)
    cluster_data(5)
    cluster_data(6)