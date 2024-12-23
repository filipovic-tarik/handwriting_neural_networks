import math
import random
import csv

import numpy as np


NUM_CLASSES = 62
TRAINING_EXAMPLES = 697_932
TEST_EXAMPLES = 1_000
K_NEIGHBORS = 5

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

# returns a list of labels for the query dataset based upon labeled observations in the dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(data,query,metric):
    # data = [list(d) for d in data]
    # for i in range(len(data)):
    #     # data[i][1] = [float(x) for x in data[i][1]]
    #     # data[i][0][0] = int(data[i][0][0])
    for i in range(len(query)):
        query[i] = [float(x) for x in query[i]]

    k = 5  
    labels = []
    # print(query,'query')
    for query_point in query:
        # print(query_point,'query_point')
        if metric == "euclidean":
            distances = [(euclidean(query_point, attrib), label[0]) for label, attrib in data]
            distances.sort(key=lambda x: x[0])  
        elif metric == "cosim":
            similarities = [(cosim(query_point, attrib), label[0]) for label, attrib in data]
            distances = sorted(similarities, key=lambda x: x[0], reverse=True)  
        
        distance_list = distances[:k]
        top_k_labels = [label for _, label in distance_list]
        print(top_k_labels,'top_k_labels')

        label_count = {}
        for label in top_k_labels:
            label_count[label] = label_count.get(label, 0) + 1
        most_common_label = max(label_count, key=label_count.get)

        labels.append(most_common_label)
    return labels

def convert_binary(data):
    # Convert all values in the matrix to floats and create a new matrix
    for i in range(len(data)):
        data[i][0] = [int(data[i][0])]
        data[i][1] = [float(x) for x in data[i][1]]

        minVal = None
        maxVal = None

        for j in range(len(data[i][1])):
            if minVal is None or data[i][1][j] < minVal:
                minVal = data[i][1][j]
            if maxVal is None or data[i][1][j] > maxVal:
                maxVal = data[i][1][j]
        for j in range(len(data[i][1])):
            data[i][1][j] = (data[i][1][j] - minVal) / (maxVal - minVal)
  
    return data 

#helper
def calculate_mean(points):
    num_points = len(points)
    num_dimensions = len(points[0])
    mean = [0] * num_dimensions  # Initialize a list for the mean of each dimension

    # Sum each dimension separately across all points
    for point in points:
        for i in range(num_dimensions):
            mean[i] += point[i]

    # Divide by the number of points to get the average (mean) position
    for i in range(num_dimensions):
        mean[i] /= num_points

    return mean




def create_confusion_matrix(actual_labels, pred_labels):
    print(actual_labels, "actual")
    print(pred_labels, "pred")
    confusion_matrix = [[0 for i in range(62)] for j in range(62)]
    for actual, pred in zip(actual_labels, pred_labels):
        confusion_matrix[actual][pred] += 1
    return confusion_matrix

def print_confusion_matrix(matrix):
    print("Confusion Matrix:")
    for row in matrix:
        print("\t".join(map(str, row)))


def label_to_character(label):
    """Convert an EMNIST ByClass label (0-61) to a readable character."""
    if 0 <= label <= 9:
        return str(label)
    if 10 <= label <= 35:
        return chr(ord('A') + label - 10)
    if 36 <= label <= 61:
        return chr(ord('a') + label - 36)
    return "?"


def read_numpy_data(file_name, limit, balanced=False):
    """Read EMNIST CSV data directly into compact NumPy arrays.

    When balanced=True, the requested number of examples is divided as evenly
    as possible among all 62 character classes. This prevents common digits
    from overwhelming less frequent letters in the training subset.
    """
    features = np.empty((limit, 784), dtype=np.float32)
    labels = np.empty(limit, dtype=np.int16)

    if balanced:
        base, remainder = divmod(limit, NUM_CLASSES)
        targets = np.full(NUM_CLASSES, base, dtype=np.int32)
        targets[:remainder] += 1
        counts = np.zeros(NUM_CLASSES, dtype=np.int32)

    accepted = 0
    with open(file_name, newline='') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            label = int(row[0])
            if balanced and counts[label] >= targets[label]:
                continue

            pixels = np.asarray(row[1:785], dtype=np.float32)
            minimum = pixels.min()
            maximum = pixels.max()
            if maximum > minimum:
                pixels = (pixels - minimum) / (maximum - minimum)
            else:
                pixels.fill(0.0)

            labels[accepted] = label
            features[accepted] = pixels
            accepted += 1
            if balanced:
                counts[label] += 1

            if accepted == limit:
                break

    if accepted < limit:
        raise ValueError(
            f"{file_name} supplied only {accepted:,} usable examples; "
            f"{limit:,} were requested."
        )

    return features, labels


def knn_numpy(train_features, train_labels, query_features, k=K_NEIGHBORS,
              batch_size=25):
    """Classify query images with k-NN using memory-efficient NumPy batches."""
    predictions = np.empty(len(query_features), dtype=np.int16)
    train_squared = np.sum(train_features * train_features, axis=1)

    for start in range(0, len(query_features), batch_size):
        stop = min(start + batch_size, len(query_features))
        queries = query_features[start:stop]
        query_squared = np.sum(queries * queries, axis=1)[:, None]

        # Squared Euclidean distance without constructing a huge
        # (queries x training examples x 784 pixels) temporary array.
        distances = (
            query_squared
            + train_squared[None, :]
            - 2.0 * queries @ train_features.T
        )

        nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
        for offset, neighbor_indices in enumerate(nearest):
            neighbor_indices = neighbor_indices[
                np.argsort(distances[offset, neighbor_indices])
            ]
            neighbor_labels = train_labels[neighbor_indices]

            counts = {}
            for label in neighbor_labels:
                label = int(label)
                counts[label] = counts.get(label, 0) + 1
            predictions[start + offset] = max(counts, key=counts.get)

        print(f"Classified {stop:,}/{len(query_features):,} test characters")

    return predictions


def print_largest_confusions(actual_labels, predicted_labels, count=10):
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    np.add.at(matrix, (actual_labels, predicted_labels), 1)
    np.fill_diagonal(matrix, 0)

    print("\nMost common mistakes:")
    flat_indices = np.argsort(matrix.ravel())[::-1]
    shown = 0
    for flat_index in flat_indices:
        occurrences = int(matrix.ravel()[flat_index])
        if occurrences == 0 or shown == count:
            break
        actual, predicted = np.unravel_index(flat_index, matrix.shape)
        print(
            f"  {label_to_character(actual)!r} predicted as "
            f"{label_to_character(predicted)!r}: {occurrences} times"
        )
        shown += 1


def read_data(file_name):
    
    data_set = []
    count = 0
    with open(file_name,'rt') as f:
        for line in f:
            if count == 1000:
                break
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
            count += 1
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')

def evaluate_knn():
    print(f"Reading all {TRAINING_EXAMPLES:,} training examples...")
    train_features, train_labels = read_numpy_data(
        'emnist-byclass-train.csv', TRAINING_EXAMPLES
    )
    print(f"Reading {TEST_EXAMPLES:,} test examples...")
    test_features, test_labels = read_numpy_data(
        'emnist-byclass-test.csv', TEST_EXAMPLES
    )

    print(
        f"Running {K_NEIGHBORS}-nearest-neighbor classification with "
        "Euclidean distance..."
    )
    predictions = knn_numpy(
        train_features, train_labels, test_features, K_NEIGHBORS
    )

    correct = int(np.sum(predictions == test_labels))
    accuracy = 100.0 * correct / len(test_labels)
    print(f"\nCorrect predictions: {correct:,}/{len(test_labels):,}")
    print(f"Accuracy: {accuracy:.1f}%")
    print_largest_confusions(test_labels, predictions)

    
def main():
    show('emnist-byclass-test.csv','pixels')


if __name__ == "__main__":
    # main()
    evaluate_knn()
    #evaluate_kmeans()
