# %% Importing the required libraries
%load_ext autoreload
%autoreload 2
import numpy as np  # Importing numpy library for numerical operations
import sys

# Importing the class and functions from the assignment
sys.path.append("../assignments/")
from assignment import *


# %% Creating an instance of the Perceptron class with learning rate 0.1 and 100 iterations
model = Perceptron(eta=0.1, n_iter=100)

# %% Generate 10 linearly separable points
X = np.array(
    [[i, i] for i in range(1, 11)]
)  # Creating an array of 10 points with 2 features each
y = np.array(
    [1 if i <= 5 else 0 for i in range(1, 11)]
)  # Creating an array of target values for the points

# %% Fitting
model.fit(X, y)  # Fitting the model to the data

# %% Testing
assert model.w_ is not None  # Checking if the weights have been initialized
assert model.b_ is not None  # Checking if the bias has been initialized

# %% Predicting
prediction = model.predict(
    [[-1, -1], [200, 200]]
)  # Predicting the class of a new point

prediction
