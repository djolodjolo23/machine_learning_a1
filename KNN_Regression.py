import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def knn_predict_y(arr):
    result = arr.sum() / arr.size
    return result


polynomial = pd.read_csv('A1_datasets/polynomial200.csv', header=None)
k_values = [1, 3, 5, 7, 9, 11]

# shuffling the dataset
n_rows = polynomial.shape[0]
indices = np.random.permutation(n_rows)
polynomial = polynomial.to_numpy()
polynomial = polynomial[indices]

# splitting the dataset into two pieces for training and testing
train_data = polynomial[:(len(polynomial) // 2), :]
test_data = polynomial[(len(polynomial)) // 2:, :]

# plotting the original data into two

'''''
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

ax1.scatter(train_data[:, 0], train_data[:, 1])
ax1.set_title('Train Data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

ax2.scatter(test_data[:, 0], test_data[:, 1])
ax2.set_title('Test Data')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

plt.show()
'''
x = train_data[:, 0]
y = train_data[:, 1]

distances = np.sqrt(np.sum((train_data[:, np.newaxis, :] - train_data) ** 2, axis=-1))
distancesx = x[:, np.newaxis] - x
indices = np.argsort(distances, axis=1)
sorted_y_values = y[indices]

# knn prediction for (f)x
results = np.zeros((len(distances), 6))
counter = 0
for k in k_values:
    closest_neighbour = []
    labels_temp = np.empty(len(distances))
    for i in range(len(train_data)):
        if k == 1:
            closest_neighbour = sorted_y_values[:, k]
        else:
            closest_neighbour = sorted_y_values[:, :k]
            prediction = knn_predict_y(closest_neighbour)
        labels_temp[i] = closest_neighbour[i] if k == 1 else prediction
    results[:, counter] = labels_temp.flatten()
    counter = counter + 1


# Get corresponding x and y values for the closest point
#nearest_y = y[nearest_indices]  # corresponding y values for the k nearest neighbors
#predicted_y = np.mean(nearest_y, axis=1)



