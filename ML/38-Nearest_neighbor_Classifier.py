import numpy as np
import pylab as pl
from sklearn import neighbors, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
Y = iris.target
print(Y)
h = .02 # step size in the mesh

knn=neighbors.KNeighborsClassifier()

# we create an instance of Neighbours Classifier and fit the data.
knn.fit(X, Y)

# Plot the decision boundary. For that, we will asign a color to each
# point in the mesh [x_min, m_max] x [y_min, y_max].
x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print(xx.shape)

new_data_predictions = knn.predict(np.c_[xx.ravel(), yy.ravel()])
print(new_data_predictions.shape)

# Put the result into a color plot
new_data_predictions = new_data_predictions.reshape(xx.shape)
print(new_data_predictions.shape)
pl.figure(1, figsize=(10, 7))
#pl.set_cmap(pl.cm.Paired)
pl.pcolormesh(xx, yy, new_data_predictions)

# Plot also the training points
pl.scatter(X[:,0], X[:,1],c = Y, s= 60  )
pl.xlabel('Sepal length')
pl.ylabel('Sepal width')

pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
pl.xticks(())
pl.yticks(())

pl.show()