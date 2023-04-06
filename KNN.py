import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def majority_vote(arr):
    count = np.bincount(arr)
    if len(count) > 1 and count[1] > count[0]:
        return 1
    else:
        return 0


def euc_distance_for_two_points(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


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

# TODO: these arrays should be filled with predicted values
TEST_labels = np.empty((0, 1))
# finding k nearest neighbors for test chips and printing OK or fail
for k in k_values:
    closest_neighbour = []
    print("K value = {}".format(k))
    for i in range(len(TEST_data)):
        coords_str = ', '.join([f"{coord: 0.1f}" for coord in TEST_data[i]])
        if k == 1:
            closest_neighbour = sorted_labels[:, k - 1]
            label = "OK" if closest_neighbour[i] == 1 else "FAIL"
        else:
            closest_neighbour = sorted_labels[:, :k]
            prediction = majority_vote(closest_neighbour[i])
            label = "OK" if prediction == 1 else "FAIL"
        new_value = np.array([[closest_neighbour[i] if k == 1 else prediction]])
        TEST_labels = np.vstack([TEST_labels, new_value])
        print(f"chip{i + 1}: [{coords_str}] ==> {label}")
print(TEST_labels)

x_min, x_max = TRAIN_data[:, 0].min() - 0.1, TRAIN_data[:, 0].max() + 0.1
y_min, y_max = TRAIN_data[:, 1].min() - 0.1, TRAIN_data[:, 1].max() + 0.1

## create meshgird with a step of 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xx_flat = xx.ravel()
yy_flat = yy.ravel()
meshgrid_array = np.column_stack((xx_flat, yy_flat))

# calculating distance for each point in the meshgrid
distances = np.sqrt(np.sum((meshgrid_array[:, np.newaxis, :] - TRAIN_data) ** 2, axis=-1))
sorted_distances = np.sort(distances, axis=1)
sorted_indices = np.argsort(distances, axis=1)
sorted_labels = TRAIN_labels[sorted_indices]

meshgrid_array_full = np.zeros((len(distances), 6))
meshgrid_array_full[:, 0] = xx_flat
meshgrid_array_full[:, 1] = yy_flat
counter = 0
# finding k nearest neighbors mesh grid points
for k in k_values:
    closest_neighbour = []
    labels_temp = np.empty(len(meshgrid_array))
    for i in range(len(meshgrid_array)):
        if k == 1:
            closest_neighbour = sorted_labels[:, k - 1]
            label = 1 if closest_neighbour[i] == 1 else 0
        else:
            closest_neighbour = sorted_labels[:, :k]
            prediction = majority_vote(closest_neighbour[i])
            label = 1 if prediction == 1 else 0
        labels_temp[i] = label
    temporary_array = labels_temp.flatten()
    meshgrid_array_full[:, counter + 2] = labels_temp.flatten()
    counter = counter + 1



# plotting 4 subplots with their decision boundary, based on the KNN for each
# point in the mesh area
start_val = 2
starting_ax = 1
x = meshgrid_array_full[:, 0]
y = meshgrid_array_full[:, 1]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
axes = [ax1, ax2, ax3, ax4]
for i, k in enumerate(k_values):
    knn = meshgrid_array_full[:, start_val]
    mask_knn_1 = knn == 1
    mask_knn_0 = knn == 0
    ax = axes[i]
    ax.scatter(x[mask_knn_1], y[mask_knn_1], c="orange", label="OK")
    ax.scatter(x[mask_knn_0], y[mask_knn_0], c="blue", label="FAIL")
    for value in set(okOrFailedVector):
        mask = value == okOrFailedVector
        ax.scatter(xdata[mask], ydata[mask], marker=markers[value])
    ax.set_title("K =" + str(k))
    ax.legend()
    start_val = start_val + 1
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()
