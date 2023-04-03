import numpy as np
import pandas as pd

microchips = pd.read_csv('A1_datasets/microchips.csv', header=None)


def euclidian_distance(train_set, test_set):
    distance = np.sqrt(np.sum((train_set - test_set) ** 2), axis=1)
    return distance

# testing this out
def _euclidian_distance(train_set, test_set):
    distances = []
    for test_point in test_set:
        row_distances = []
        for train_point in train_set:
            distance = np.sqrt(np.sum((test_point - train_point) ** 2))
            row_distances.append(distance)
        distances.append(row_distances)
    return np.array(distances)


def predict(train_set, train_labels, test_set, k):
    distance = _euclidian_distance(train_set, test_set)
    nearest_indices = np.argsort(distance)[:k]
    nearest_labels = train_labels[nearest_indices]
    return np.bincount(nearest_labels).argmax()


TRAIN_data = np.array(microchips.iloc[:, :2].values)
TRAIN_labels = np.array(microchips.iloc[:, 2].values)
test_chips = np.array([[-0.3, 1.0],
                       [-0.5, -0.1],
                       [0.6, 0.0]])
k_values = [1, 3, 5, 7]

for i, train_data in enumerate(TRAIN_data):
    print("Chip {}:".format(i + 1))
    for k in k_values:
        class_label = predict(TRAIN_data, TRAIN_labels, test_chips, k)
        print("  k = {}: {}".format(k, "OK" if class_label == 1 else "Fail"))
