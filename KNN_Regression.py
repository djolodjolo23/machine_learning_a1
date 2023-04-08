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
'''


results = np.zeros((len(x), 6))
counter = 0
for k in k_values:
    closest_neighbour = []
    labels_temp = np.empty(100)
    for i in range(len(x)):
        num = x[i-1]
        distance_test = np.abs(x - num)
        sorted_indices = distance_test.argsort()
        if k == 1:
            closest_indices = sorted_indices[k]
            closest_x = x[closest_indices]
            closest_neighbour = y[closest_indices]
        else:
            closest_indices = sorted_indices[1:k+1]
            closest_x = x[closest_indices]
            closest_neighbour = y[closest_indices]
            prediction = knn_predict_y(closest_neighbour)
        labels_temp[i] = closest_neighbour if k == 1 else prediction
    results[:, counter] = labels_temp.flatten()
    counter = counter + 1


# sorting the y values for k = 1
ydata = results[:, 0] # here 0 determines the first column in the result array for k=1
sorted_y = [ydata[i] for i in sorted_indices]
sorted_x = [x[i] for i in sorted_indices]

#slope, intercept, r_value, p_value, std_err = linregress()

# plot the line
#plt.plot(sorted_x, sorted_y)
plt.scatter(x, y, color='blue', label='Points')

# add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial curve through x and y coordinates')

# display the plot
plt.show()
'''


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


train_distances = knn(k_values, train_data, train_data)
test_distances = knn(k_values, test_data, test_data)


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
for i, k in enumerate(k_values):
    ax = axes[i]
    x_train = x
    y_train = train_distances[:, i]
    x_test = test_data[:, 0]
    y_test = test_data[:, 1]
    sort_idx = np.argsort(x_train)
    x_train_sorted = x_train[sort_idx]
    y_train_sorted = y_train[sort_idx]
    degree = 100
    z = np.polyfit(x, y_train, degree)
    p = np.poly1d(z)
    y_fit = p(x_train_sorted)
    y_predicted = np.polyval(p, x_test)
    mse = mean_squared_error(y_test, y_predicted)
    window_size = 2
    y_fit_smooth = savgol_filter(y_fit, window_size, 1)
    ax.plot(x_train_sorted, y_train_sorted, 'o')
    ax.plot(x_train_sorted, y_fit_smooth, label='Fitted function')
    ax.set_title("K = " + str(k) + " , MSE: " + str(mse))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()
