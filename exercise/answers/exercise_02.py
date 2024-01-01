#
# In this expercise, you will:
# 1. learn how to perform feature selection with L1 regularization
# 2. learn how to measure the feature importance with SHAP
# %%
import numpy as np
import shap
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# %% Load the data:
# This will import the data_table implemented in the previous exercise
from exercise_01 import *

# %% TODO: Perform feature selection with L1 regularization
# Instructions:
# 1. Define the target variable "y" as "Survived" column from the data_table.
# 2. Define the features "X" as all columns from the data_table except "Survived", "Name", "Ticket", "Cabin".
# 3. Create a LogisticRegression model with L1 penalty.
# 4. Fit the model with the features and target variable.
focal_features = [
    col
    for col in data_table.columns
    if col not in ["Survived", "Name", "Ticket", "Cabin"]
]

y = data_table["Survived"].values
X = data_table[focal_features].values

model = LogisticRegression(penalty="l1", solver="liblinear", random_state=42, C=1)
model.fit(X, y)


# %% TODO: See the regression coefficient learned by the model
model.coef_

# %% TODO: Draw the lasso path of the model with L1 penalty from C = 0.001 to 10, with 30 samples
C_list = np.logspace(-3, 2, 30)

coefs = []
for C in C_list:
    model = LogisticRegression(penalty="l1", solver="liblinear", C=C, random_state=42)
    model.fit(X, y)
    coefs.append(model.coef_[0])

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(8, 5))

plot_data = pd.DataFrame(coefs, columns=focal_features)
plot_data["C"] = C_list

# Melt the data for plotting
plot_data = plot_data.melt(id_vars="C", var_name="Feature", value_name="Coefficient")

# Plot the data
sns.lineplot(
    data=plot_data,
    x="C",
    y="Coefficient",
    hue="Feature",
    palette=sns.color_palette(),
    ax=ax,
)

ax.set_xlabel("C")
ax.set_xscale("log")
ax.set_ylabel("Coefficients")
ax.set_title("Lasso paths")
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, shadow=True, ncol=4
)

sns.despine()


# %%
