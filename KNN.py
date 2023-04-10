import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def classify_x_test_and_print(k_values):
    for k_value in k_values:
        print("K value = {}".format(k_value))
        for i in range(len(TEST_data)):
            coords_str = ', '.join([f"{coord: 0.1f}" for coord in TEST_data[i]])
            if k_value == 1:
                closest_neighbour = sorted_labels[:, k_value - 1]
                label = "OK" if closest_neighbour[i] == 1 else "FAIL"
            else:
                closest_neighbour = sorted_labels[:, :k_value]
                prediction = majority_vote(closest_neighbour[i])
                label = "OK" if prediction == 1 else "FAIL"
            print(f"chip{i + 1}: [{coords_str}] ==> {label}")

def knn(k_values, minuend_data, subtrahend_data):
    distances = np.sqrt(np.sum((minuend_data[:, np.newaxis, :] - subtrahend_data) ** 2, axis=-1))
    sorted_indices = np.argsort(distances, axis=1)
    sorted_labels = TRAIN_labels[sorted_indices]
    array_full = np.zeros((len(distances), 6))
    array_full[:, 0] = minuend_data[:, 0]
    array_full[:, 1] = minuend_data[:, 1]
    counter = 0
    for k in k_values:
        labels_temp = np.empty(len(distances))  # this needs to be changed
        for i in range(len(distances)):
            if k == 1:
                closest_neighbour = sorted_labels[:, k - 1]
                label = 1 if closest_neighbour[i] == 1 else 0
            else:
                closest_neighbour = sorted_labels[:, :k]
                prediction = majority_vote(closest_neighbour[i])
                label = 1 if prediction == 1 else 0
            labels_temp[i] = label
        array_full[:, counter + 2] = labels_temp.flatten()
        counter = counter + 1
    return array_full


def majority_vote(arr):
    count = np.bincount(arr)
    if len(count) > 1 and count[1] > count[0]:
        return 1
    else:
        return 0


microchips = pd.read_csv('A1_datasets/microchips.csv', header=None)

xdata = microchips.iloc[:, 0]
ydata = microchips.iloc[:, 1]
okOrFailedVector = microchips.iloc[:, 2]

markers = {1: 'o', 0: 'x'}
for value in set(okOrFailedVector):
    mask = value == okOrFailedVector
    plt.scatter(xdata[mask], ydata[mask], marker=markers[value], label=value)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title='Value')
plt.show()

TRAIN_data = np.array(microchips.iloc[:, :2].values)
TRAIN_labels = np.array(microchips.iloc[:, 2].values)
TEST_data = np.array([[-0.3, 1.0],
                      [-0.5, -0.1],
                      [0.6, 0.0]])
k_values = [1, 3, 5, 7]

distances = np.sqrt(np.sum((TEST_data[:, np.newaxis, :] - TRAIN_data) ** 2, axis=-1))
# sorted_distances = np.sort(distances, axis=1)
sorted_indices = np.argsort(distances, axis=1)
sorted_labels = TRAIN_labels[sorted_indices]

classify_x_test_and_print(k_values)

x_min, x_max = TRAIN_data[:, 0].min() - 0.1, TRAIN_data[:, 0].max() + 0.1
y_min, y_max = TRAIN_data[:, 1].min() - 0.1, TRAIN_data[:, 1].max() + 0.1

## create meshgird with a step of 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xx_flat = xx.ravel()
yy_flat = yy.ravel()
meshgrid_array = np.column_stack((xx_flat, yy_flat))

# this was meshgrid array full
full_array = knn(k_values, meshgrid_array, TRAIN_data)

# plotting 4 subplots with their decision boundary, based on the KNN for each
# point in the mesh area
distances_train_data = knn(k_values, TRAIN_data, TRAIN_data)

start_val = 2
starting_ax = 1
x = full_array[:, 0]
y = full_array[:, 1]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
axes = [ax1, ax2, ax3, ax4]
for i, k in enumerate(k_values):
    knn = full_array[:, start_val]
    mask_knn_1 = knn == 1
    mask_knn_0 = knn == 0
    ax = axes[i]
    diff_array = np.not_equal(TRAIN_labels, distances_train_data[:, start_val])
    num_diff = np.sum(diff_array)
    ax.scatter(x[mask_knn_1], y[mask_knn_1], c="orange", label="OK")
    ax.scatter(x[mask_knn_0], y[mask_knn_0], c="blue", label="FAIL")
    for value in set(okOrFailedVector):
        mask = value == okOrFailedVector
        ax.scatter(xdata[mask], ydata[mask], marker=markers[value])
    ax.set_title("K =" + str(k) + " ,Training errors: "
                 + str(num_diff))
    ax.legend()
    start_val = start_val + 1
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()
