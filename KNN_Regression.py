import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def knn_predict_y(arr):
    all_y = arr[:, 1]
    result = all_y.sum() / all_y.size
    return result


def calculate_regression(k_value):
    points_regression = np.empty(shape=(0, 2))
    for coordinates in test_data:
        current_x = coordinates[0]
        predicted_point = predict(current_x, k_value)
        points_regression = np.append(points_regression, [predicted_point], axis=0)
    return points_regression


def predict(current_x, k_value):
    distance = np.abs(x - current_x)
    sorted_indices = np.argsort(distance)
    points = np.empty(shape=(0, 2))
    for k_value in range(k_value):
        closest_point_index = sorted_indices[k_value]
        point = (x[closest_point_index], y[closest_point_index])
        points = np.append(points, [[point[0], point[1]]], axis=0)
    predicted_y = knn_predict_y(points)
    predicted_point = np.array([current_x, predicted_y])
    return predicted_point


polynomial = pd.read_csv('A1_datasets/polynomial200.csv', header=None)
k_values = [1, 3, 5, 7, 9, 11]

# shuffling the dataset
n_rows = polynomial.shape[0]
indices = np.random.permutation(n_rows)
polynomial = polynomial.to_numpy()
polynomial = polynomial[indices]

train_data, test_data = np.split(polynomial, 2)

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

#
x_train = x = train_data[:, 0]
y_train = y = train_data[:, 1]


x_test = test_data[:, 0]
y_test = test_data[:, 1]
sort = np.argsort(x_test)
x_test_sorted = x_test[sort]
y_test_sorted = y_test[sort]


sort_idx = np.argsort(x_train)
x_train_sorted = x_train[sort_idx]
y_train_sorted = y_train[sort_idx]

'''''
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
for i, k in enumerate(k_values):
    ax = axes[i]
    r = calculate_regression(k)
    mse = np.mean(np.square(y_test - r[:, 1]))
    rounded_mse = np.round(mse, decimals=2)
    sorted_regression_data = r[r[:, 0].argsort()]
    x_regression = np.array(sorted_regression_data[:, 0])
    y_regression = np.array(sorted_regression_data[:, 1])
    ax.plot(x_train_sorted, y_train_sorted, 'o')
    ax.plot(x_regression, y_regression, 'r-')
    ax.set_title("K = " + str(k) + " , MSE: " + str(rounded_mse))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()
'''