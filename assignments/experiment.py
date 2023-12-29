# %% Load the penguin data
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from assignment import *

# %% Load the penguin data
X, labels = load_penguin_data()  # Load the penguin data from the assignment module

# %% Let's focus on two species: Adelie and Chinstrap
focal_species = ["Adelie", "Chinstrap"]  # Define the species we are interested in
focal_data = np.isin(labels, focal_species)  # Filter the data for these species
X, labels = X[focal_data], labels[focal_data]  # Apply the filter to our data

# %% Convert the species to numerical values
species, y = np.unique(
    labels, return_inverse=True
)  # Convert species names to numerical values

# %% TODO: Creating an instance of the Perceptron class

# %% TODO: Draw the decision regions by using `plot_decision_regions` function from utils.py. Label the axes with "bill_length_mm" and "bill_depth_mm".
from utils import plot_decision_regions  # Import the function to plot decision regions

# %% TODO Save the figure as png


# %% Optional: Fit Adaline and draw the decision regions.


# %% Optional: Apply the standard scaler to the data and draw the decision regions.
