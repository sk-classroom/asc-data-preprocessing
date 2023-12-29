# %%
# In this expercise, you will:
# 3. learn how to use the LogisticRegression class from the scikit-learn library.
# 4. learn how to evaluate the performance of the model using different metrics.
# 4. learn how to use the plot_decision_regions function from the utils.py module.
# %%
%load_ext autoreload
%autoreload 2
import sys
import numpy as np
sys.path.append("../assignments")
from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from utils import plot_decision_regions
import matplotlib.pyplot as plt

# Load the penguin dataset
penguins = sns.load_dataset("penguins")

# Drop the rows with missing values
penguins = penguins.dropna()

penguins.describe()
# %%
# We will use the bill length and depth as features
# and the species as the target variable
X = penguins[["bill_length_mm", "bill_depth_mm"]].values
y = penguins["species"].values
# %% TODO: Split the data into training and testing sets. Use the following parameters:
# - test_size=0.2
# - random_state=42
# - stratify=y
# Use the train_test_split function from the scikit-learn library
X_train, X_test, y_train, y_test = ...

# TODO: Fit the logistic regression model using scikit-learn using the following parameters:
# - penalty=None # The penalty term is used to prevent overfitting. The "none" means no penalty term
# - solver = "saga" # The solver for weight optimization. "sag" refers to Stochastic Average Gradient descent
# - multi_class="ovr"  # The "ovr" stands for One-vs-Rest, which means that in the case of multi-class classification, a separate model is trained for each class predicted against all other classes
# - random_state=42  # The seed used by the random number generator for shuffling the data
# Use the LogisticRegression class from the scikit-learn library
lr = ...

lr.fit(...)

# %% Plot the decision regions of the trained Logistic Regression classifier
X_combined = np.vstack([X_train, X_test])
y_combined = np.hstack((y_train, y_test))
test_idx = range(len(y_train), len(y_train) + len(y_test))
plot_decision_regions(
    X_combined,  # The combined feature matrix of training and testing sets
    y_combined,  # The combined target variable of training and testing sets
    classifier=lr,  # The trained Logistic Regression classifier
    test_idx=test_idx  # The indices of the test set examples
)
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.title('Logistic Regression Decision Regions')

# %% TODO: Evaluate the performance by using accuracy as a metric. Do not use scikit-learn's accuracy_score function. Use numpy to calculate the accuracy.
y_pred = lr.predict(...)
acc = ...
print("Accuracy: %f" % acc)