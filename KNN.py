import numpy as np
import pandas as pd


def majority_vote(arr):
    count = np.bincount(arr)
    if count[1] > count[0]:
        return 1
    else:
        return 0

microchips = pd.read_csv('A1_datasets/microchips.csv', header=None)

TRAIN_data = np.array(microchips.iloc[:, :2].values)
TRAIN_labels = np.array(microchips.iloc[:, 2].values)
test_chips = np.array([[-0.3, 1.0],
                       [-0.5, -0.1],
                       [0.6, 0.0]])
k_values = [1, 3, 5, 7]

distances = np.sqrt(np.sum((test_chips[:, np.newaxis, :] - TRAIN_data)**2, axis=-1))
sorted_distances = np.sort(distances, axis=1)
sorted_indices = np.argsort(distances, axis=1)
sorted_labels = TRAIN_labels[sorted_indices]


for k in k_values:
    closest_neighbour = []
    print("K value = {}".format(k))
    for i in range(len(test_chips)):
        coords_str = ', '.join([f"{coord: 0.1f}" for coord in test_chips[i]])
        if k == 1:
            closest_neighbour = sorted_labels[:, k-1]
            label = "OK" if closest_neighbour[i] == 1 else "FAIL"
        else:
            closest_neighbour = sorted_labels[:, :k]
            prediction = majority_vote(closest_neighbour[i])
            label = "OK" if prediction == 1 else "FAIL"
        print(f"chip{i + 1}: [{coords_str}] ==> {label}")
