#
# In this expercise, you will:
# 1. learn the effect of feature scaling on the performance of classification
# 2. learn how to perform scale normalization by using the scikit-learn library to standardize features.
# %%
%load_ext autoreload
%autoreload 2
import sys
import numpy as np
sys.path.append("../assignments")
from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
focal_cols = [
    "bill_length_mm",
    "body_mass_g",
]
X = penguins[focal_cols].values
y = penguins["species"].values

# %% Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,  # Feature matrix
    y,  # Target variable
    test_size=0.2,  # Proportion of the dataset to include in the test split
    random_state=42,  # The seed used by the random number generator
    stratify=y  # This stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify
)

# %% Apply the logistic regression model using scikit-learn without regularization.
lr = LogisticRegression(
    penalty = None, # The penalty term is used to prevent overfitting. The "none" means no penalty term
    solver = "saga",
    multi_class="ovr",  # The "ovr" stands for One-vs-Rest, which means that in the case of multi-class classification, a separate model is trained for each class predicted against all other classes
    random_state=42  # The seed used by the random number generator for shuffling the data
)
lr.fit(
    X_train,  # The training data
    y_train  # The target variable to try to predict in the case of supervised learning
)

# %% Evaluate the performance by using accuracy as a metric. Do not use scikit-learn's accuracy_score function. Use numpy to calculate the accuracy.
y_pred = lr.predict(X_test)
acc = np.mean(y_pred == y_test)
print("Accuracy: %f" % acc)

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


# %% TODO: Use the StandardScaler class from the scikit-learn library to standardize features
...

# %% TODO: Apply the logistic regression model using scikit-learn. Use the same parameters as the TODO cell above.
lr_std = ...

# %% TODO: Evaluate the accuracy of the prediction for the test data
acc =
print("Accuracy: %f" % acc)

# %% Plot the decision regions of the trained Logistic Regression classifier
X_combined_std = ...
test_idx = range(len(y_train), len(y_train) + len(y_test))
plot_decision_regions(
    X_combined_std,  # The combined feature matrix of training and testing sets
    y_combined,  # The combined target variable of training and testing sets
    classifier=lr_std,  # The trained Logistic Regression classifier
    test_idx=test_idx  # The indices of the test set examples
)
plt.xlabel('Rescaled Bill Length (mm)')
plt.ylabel('Rescaled Bill Depth (mm)')
plt.title('Logistic Regression Decision Regions')