import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import  pyplot as plt

x = 5 * np.random.rand(40, 1)
print(x.shape)

y = np.sin(x).ravel() #flattens the array

# add noise to every 5th element

y[::5]  += 1 * (0.5 - np.random.rand(8))

# The points we want to predict

T = np.linspace(0, 5, 50)[:, np.newaxis]
print(T.shape)

# now lets fit the model

# how many neigbors to use for regression
n_neighbors = 5
k = 1
for i, weights in enumerate(['uniform', 'distance']):
    knn = KNeighborsRegressor(n_neighbors, weights = weights)
    fitted_model = knn.fit(x, y)
    predictions = fitted_model.predict(x)
    predictions_new_data = fitted_model.predict(T)

    plt.subplot(2, 2, k)
    plt.scatter(x, y, c='k', label='data')
    plt.scatter(x, predictions, c='r', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,  weights))

    plt.subplot(2, 2, k + 1)
    plt.scatter(x, y, c='k', label='data')
    plt.plot(T, predictions_new_data, c='r', label='prediction')

    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,  weights))

    k += 2

plt.show()