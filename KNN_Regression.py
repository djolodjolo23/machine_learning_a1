import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error


def knn_predict_y(arr):
    result = arr.sum() / arr.size
    return result



polynomial = pd.read_csv('A1_datasets/polynomial200.csv', header=None)
k_values = [1, 3, 5, 7, 9, 11]

# shuffling the dataset
#n_rows = polynomial.shape[0]
#indices = np.random.permutation(n_rows)
polynomial = polynomial.to_numpy()
#polynomial = polynomial[indices]

# splitting the dataset into two pieces for training and testing
#train_data = polynomial[:(len(polynomial) // 2), :]
#test_data = polynomial[(len(polynomial)) // 2:, :]
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

x = train_data[:, 0]
y = train_data[:, 1]



def knn(k_values, minuend_data, subtrahend_data):
    distances = np.sqrt(np.sum((minuend_data[:, np.newaxis, :] - subtrahend_data) ** 2, axis=-1))
    array_full = np.zeros((len(distances), 6))
    counter = 0
    for k in k_values:
        temp = np.empty(len(distances))
        for i in range(len(distances)):
            closest = np.argsort(distances[i])
            if k == 1:
                closest_neighbour = y[closest]
            else:
                closest_neighbour = y[closest]
                k_closest_neighbours = closest_neighbour[:k]
                prediction = knn_predict_y(k_closest_neighbours)
            temp[i] = closest_neighbour[1] if k == 1 else prediction
        array_full[:, counter] = temp.flatten()
        counter = counter + 1
    return array_full

def calculateMSE(data, regression_data):
    y_data = np.sort(data)
    y_regression = np.sort(regression_data)
    mse = 0
    data_length = len(y_data)
    for i in range(data_length):
        y_t = y_data[i]
        y_r = y_regression[i]
        mse += ((y_t-y_r)**2)
    return mse / data_length


def calculateKNNRegression(k):

    # Predict points using kNN-regression
    points_regression = np.empty(shape=(0, 2))
    counter = 0
    for xy_test in test_data:
        test_x = xy_test[0]
        predicted_point = predictPoint(train_data, test_x, k)
        points_regression = np.append(points_regression, [predicted_point], axis=0)
        counter = counter + 1
    return points_regression

def predictPoint(train_data, x_test, k):
    # All distances to the points in train data
    # from a single x point in test data
    x = train_data[:, 0]
    y = train_data[:, 1]
    distance = np.abs(x - x_test)
    sorted_indices = np.argsort(distance)
    # Get k-amount of shortest piints
    points = np.empty(shape=(0, 2))
    for i in range(k):
        closest_point_index = sorted_indices[i]
        point = (x[closest_point_index], y[closest_point_index])
        points = np.append(points, [[point[0], point[1]]], axis=0)
    # Vote to get the most approximate point
    predicted_y = 0
    for point in points:
        y = point[1]
        predicted_y += y
    predicted_y = predicted_y / k

    # Predicted point
    predicted_point = np.array([x_test, predicted_y])

    return predicted_point

    # Returns a list of distances which are sorted from shortest to longest
    # [distance, index]
    # distance - flaot value of distance
    # index - index of the data point in train_data that maps to that distance value


    # Method


train_distances = knn(k_values, train_data, train_data)
test_distances = knn(k_values, test_data, train_data)

x_test = test_data[:, 0]
y_test = test_data[:, 1]
sort = np.argsort(x_test)
x_test_sorted = x_test[sort]
y_test_sorted = y_test[sort]

x_train = x
y_train = y
sort_idx = np.argsort(x_train)
x_train_sorted = x_train[sort_idx]
y_train_sorted = y_train[sort_idx]

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
for i, k in enumerate(k_values):
    ax = axes[i]

    #x_train = x
    #y_train = y


    #sort_idx = np.argsort(x_train)
    #x_train_sorted = x_train[sort_idx]
    #y_train_sorted = y_train[sort_idx]

    r = calculateKNNRegression(k)


    mse = calculateMSE(y_train, r)
    #mse = round(mse, 2)
    sorted_regression_data = r[r[:, 0].argsort()]
    x_regression = np.array(sorted_regression_data[:, 0])
    y_regression = np.array(sorted_regression_data[:, 1])
    ax.plot(x_train_sorted, y_train_sorted, 'o')
    ax.plot(x_regression, y_regression, 'r-')
    ax.set_title("K = " + str(k) + " , MSE: " + str(mse))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()
