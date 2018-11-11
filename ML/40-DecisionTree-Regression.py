# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import  tree
import graphviz

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort( 5 * rng.rand(80, 1), axis=0)
print(X.shape)
y = np.sin(X).ravel() # Flatten the array.
y[::5] += 3 * (0.5 - rng.rand(16))
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
fig = plt.figure(1, figsize=(8, 6))
ax =fig.add_subplot(1, 1, 1)
ax.scatter(X, y, s= 60, edgecolor="black", c= "darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

dot_data = tree.export_graphviz(regr_1,out_file=None,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("sin-regression-depth2")

dot_data = tree.export_graphviz(regr_2,out_file=None,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("sin-regression-depth 5")
