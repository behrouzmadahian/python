"""
categorical_accuracy checks to see if the index of the maximal true value is equal to the index of
the maximal predicted value.
Y_true MUST be one_ hot!

sparse_categorical_accuracy checks to see if the maximal true value is equal
to the index of the maximal predicted value.
the categorical_accuracy corresponds to a one-hot encoded vector for y_true
Y_true can be integer corr to correct class!

categorical_cross_entropy:
target is one hot encoded

sparse_categorical_cross_entropy:
target is integer
"""
