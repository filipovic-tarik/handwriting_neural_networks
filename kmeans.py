import math
import random

# returns Euclidean distance between vectors and b
def euclidean(a,b):
    squared_diffs = 0
    for i in range(len(a)):
        squared_diffs += (a[i] - b[i]) ** 2
    
    dist = math.sqrt(squared_diffs)
    return(dist)
        
# returns Cosine Similarity between vectors and b
def cosim(a,b):
    dot_product = 0
    a_mag = 0
    b_mag = 0
    
    for i in range (len(a)):
        dot_product += a[i] * b[i] 
        a_mag += math.pow(a[i], 2)
        b_mag += math. pow(b[i],2)
    a_mag = math.sqrt(a_mag)
    b_mag = math.sqrt(b_mag)
    if a_mag == 0 or b_mag == 0:
        return 0.0
    dist = dot_product/(a_mag*b_mag)
    return(dist)

# returns Hamming distance between vectors and b
def hamming(a,b):
    dist = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            dist += 1
    return(dist)

# returns Pearson Correkation between vectors and b
def pearson(a,b):
    n = len(a)
    a_sum = sum(a)
    b_sum = sum(b)
    a_sum_squared = sum(x ** 2 for x in a)
    b_sum_squared = sum(y ** 2 for y in b)
    sum_ab = sum(x*y for x, y in zip(a,b))
    numerator = (n * sum_ab) - (a_sum * b_sum)
    denominator = ((n * a_sum_squared - a_sum ** 2) * (n * b_sum_squared - b_sum ** 2)) ** 0.5      
    if denominator == 0:
        return 0.0
    dist = numerator/denominator

    return(dist)

def initialize_centroids(data, k):
    random_indices = random.sample(range(len(data)), k)
    centroids = [data[i] for i in random_indices]
    return centroids

def assign_clusters(data, centroids, metric):
    clusters = [[] for _ in range(len(centroids))]
    for idx, point in enumerate(data):
        if metric == "euclidean":
            distances = [euclidean(point, centroid) for centroid in centroids]
        elif metric == "cosim":
            distances = [cosim(point, centroid) for centroid in centroids]
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosim'.")
        
        nearest_centroid_index = distances.index(min(distances))
        clusters[nearest_centroid_index].append(idx)  # Store indices
    return clusters

def update_centroids(clusters, data):
    centroids = []
    for cluster in clusters:
        if len(cluster) > 0:
            num_dimensions = len(data[0])
            new_centroid = [0] * num_dimensions
            for idx in cluster:
                point = data[idx]
                for i in range(num_dimensions):
                    new_centroid[i] += point[i]
            num_points = len(cluster)
            for i in range(num_dimensions):
                new_centroid[i] /= num_points
        else:
            new_centroid = [0] * len(data[0])  
        centroids.append(new_centroid)
    return centroids

def kmeans(data, data_labels, query, metric, k=10, max_iterations=100):
    centroids = initialize_centroids(data, k)
    
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids, metric)
        
        new_centroids = update_centroids(clusters, data)
        
        if centroids == new_centroids:
            break
        
        centroids = new_centroids
    
    cluster_label_map = {}
    for cluster_idx, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
        labels_in_cluster = [data_labels[idx] for idx in cluster]
        label_counts = {}
        for label in labels_in_cluster:
            label_counts[label] = label_counts.get(label, 0) + 1
        most_common_label = max(label_counts, key=label_counts.get)
        cluster_label_map[cluster_idx] = most_common_label
    
    labels = []
    for query_point in query:
        if metric == "euclidean":
            distances = [euclidean(query_point, centroid) for centroid in centroids]
        elif metric == "cosim":
            distances = [cosim(query_point, centroid) for centroid in centroids]
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosim'.")
        
        nearest_centroid_index = distances.index(min(distances))
        predicted_label = cluster_label_map.get(nearest_centroid_index, -1)  # Use -1 if cluster not found
        labels.append(predicted_label)
    
    return labels