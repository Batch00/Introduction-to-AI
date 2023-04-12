import csv
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt


def load_data(filepath):
    with open(filepath, 'r') as f:
        dict_reader = csv.DictReader(f)
        list_of_dict = list(dict_reader)
        return list_of_dict

def calc_features(row):
    x1 = int(row['Attack'])
    x2 = int(row['Sp. Atk'])
    x3 = int(row['Speed'])
    x4 = int(row['Defense'])
    x5 = int(row['Sp. Def'])
    x6 = int(row['HP'])
    return np.array([x1,x2,x3,x4,x5,x6], dtype=np.int64)

def hac(features):
    n = len(features)

    # calculate distance matrix
    X = np.array(features)
    p1 = np.sum(np.array(X)**2, axis=1).reshape(-1,1)
    p2 = np.sum(np.array(X)**2, axis=1)
    p3 = -2 * np.dot(np.array(X), np.array(X).T)
    dist_mat = np.sqrt(p1 + p2 + p3)

    # Initialize clusters
    clusters = [[i] for i in range(n)]
    not_exist = []
    all_merged = []
    # Initialize indices of clusters
    cluster_indices = list(range(n))

    # Initialize linkage matrix
    Z = np.zeros((n-1, 4))

    # Loop through all iterations
    for i in range(len(Z)):
        # Find the two clusters with minimum distance
        min_dist = np.inf
        for j in range(len(clusters)):
            for k in range(j+1, len(clusters)):
                dist = np.max(dist_mat[clusters[j],:][:,clusters[k]])
                
                if j in not_exist or k in not_exist:
                    continue
                if dist < min_dist:
                    min_dist = dist
                    merge_clusters = [j, k]

        # Update clusters
        new_cluster = clusters[merge_clusters[0]] + clusters[merge_clusters[1]]

        all_merged.append((new_cluster, i))
        not_exist.append(merge_clusters[0])
        not_exist.append(merge_clusters[1])
        clusters.append(new_cluster)
        # Update indices
        cluster_indices.append(n+i)
        
        # Update linkage matrix column 3 and 4
        Z[i, 2] = min_dist
        Z[i, 3] = len(new_cluster)
        
        # Update linkage matrix column 1 and 2
        if len(new_cluster) == 2:
            Z[i, 0] = new_cluster[0]
            Z[i, 1] = new_cluster[1]
        else:
            if (len(clusters[merge_clusters[0]]) < 2):
                Z[i,0] = merge_clusters[0]
            else:
                Z[i,0] = n + all_merged[([y[0] for y in all_merged].index(clusters[merge_clusters[0]]))][1]
            
            if (len(clusters[merge_clusters[1]]) < 2):
                Z[i,1] = merge_clusters[1]
            else:
                Z[i,1] = n + all_merged[([y[0] for y in all_merged].index(clusters[merge_clusters[1]]))][1]

    return Z

def imshow_hac(Z, names):
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z, labels=names, leaf_rotation=90)
    fig.tight_layout()
    return plt.show()