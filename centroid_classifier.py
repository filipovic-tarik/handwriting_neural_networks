import math
import random


# Returns Euclidean distance between vectors a and b
def centroid_euclidean_distance(a, b):
    squared_diffs = 0
    for i in range(len(a)):
        squared_diffs += (a[i] - b[i]) ** 2

    dist = math.sqrt(squared_diffs)
    return dist


# Returns cosine similarity between vectors a and b
def centroid_cosine_similarity(a, b):
    dot_product = 0
    a_mag = 0
    b_mag = 0

    for i in range(len(a)):
        dot_product += a[i] * b[i]
        a_mag += math.pow(a[i], 2)
        b_mag += math.pow(b[i], 2)
    a_mag = math.sqrt(a_mag)
    b_mag = math.sqrt(b_mag)
    if a_mag == 0 or b_mag == 0:
        return 0.0
    dist = dot_product / (a_mag * b_mag)
    return dist


# Returns Hamming distance between vectors a and b
def centroid_hamming_distance(a, b):
    dist = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            dist += 1
    return dist


# Returns Pearson correlation between vectors a and b
def centroid_pearson_correlation(a, b):
    n = len(a)
    a_sum = sum(a)
    b_sum = sum(b)
    a_sum_squared = sum(x ** 2 for x in a)
    b_sum_squared = sum(y ** 2 for y in b)
    sum_ab = sum(x * y for x, y in zip(a, b))
    numerator = (n * sum_ab) - (a_sum * b_sum)
    denominator = (
        (n * a_sum_squared - a_sum ** 2)
        * (n * b_sum_squared - b_sum ** 2)
    ) ** 0.5
    if denominator == 0:
        return 0.0
    dist = numerator / denominator

    return dist


def choose_initial_centroids(data, k):
    random_indices = random.sample(range(len(data)), k)
    centroids = [data[i] for i in random_indices]
    return centroids


def group_by_centroid(data, centroids, metric):
    clusters = [[] for _ in range(len(centroids))]
    for idx, point in enumerate(data):
        if metric == "euclidean":
            distances = [centroid_euclidean_distance(point, centroid) for centroid in centroids]
        elif metric == "cosim":
            distances = [centroid_cosine_similarity(point, centroid) for centroid in centroids]
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosim'.")

        if metric == "euclidean":
            nearest_centroid_index = distances.index(min(distances))
        else:
            nearest_centroid_index = distances.index(max(distances))
        clusters[nearest_centroid_index].append(idx)  # Store indices
    return clusters


def recalculate_centroids(clusters, data):
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


def classify_with_centroids(data, data_labels, query, metric, k=10, max_iterations=100):
    centroids = choose_initial_centroids(data, k)

    for _ in range(max_iterations):
        clusters = group_by_centroid(data, centroids, metric)

        new_centroids = recalculate_centroids(clusters, data)

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
            distances = [
                centroid_euclidean_distance(query_point, centroid)
                for centroid in centroids
            ]
        elif metric == "cosim":
            distances = [
                centroid_cosine_similarity(query_point, centroid)
                for centroid in centroids
            ]
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosim'.")

        if metric == "euclidean":
            nearest_centroid_index = distances.index(min(distances))
        else:
            nearest_centroid_index = distances.index(max(distances))
        # Use -1 when the nearest centroid has no assigned label
        predicted_label = cluster_label_map.get(nearest_centroid_index, -1)
        labels.append(predicted_label)

    return labels
