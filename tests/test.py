import unittest
import numpy as np
import sys
import pandas as pd

sys.path.append("assignments/")
from assignment import *
from scipy import stats


class TestAssignment(unittest.TestCase):
    def setUp(self):
        self.dtypes = {
            "PassengerId": "int64",
            "Survived": "int64",
            "Pclass": "str",
            "Name": "str",
            "Sex": "str",
            "Age": "float64",
            "SibSp": "int64",
            "Parch": "int64",
            "Ticket": "str",
            "Fare": "float64",
            "Cabin": "str",
            "Embarked": "str",
        }
        self.data_loader = DataLoader(
            path="data/train.csv",
            dtypes=self.dtypes,
            nominal=["Sex", "Embarked"],
            ordinal={"Pclass": {"1": 1, "2": 2, "3": 3}},
            target="Survived",
            drop=["Name", "Ticket", "Cabin"],
        )
        self.Cs = np.logspace(-4, 4, 10)

    def test_data_loader(self):
        X, y, feature_names = self.data_loader.load()
        df = pd.read_csv("tests/data.csv")
        np.testing.assert_array_almost_equal(
            X.astype(float), df[feature_names].values.astype(float), decimal=2
        )
        np.testing.assert_array_almost_equal(
            y.astype(float), df["target"].values.astype(float), decimal=5
        )


class TestClassificationLassoPath(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader(
            path="data/train.csv",
            dtypes={
                "PassengerId": "int64",
                "Survived": "int64",
                "Pclass": "str",
                "Name": "str",
                "Sex": "str",
                "Age": "float64",
                "SibSp": "int64",
                "Parch": "int64",
                "Ticket": "str",
                "Fare": "float64",
                "Cabin": "str",
                "Embarked": "str",
            },
            nominal=["Sex", "Embarked"],
            ordinal={"Pclass": {"1": 1, "2": 2, "3": 3}},
            target="Survived",
            drop=["Name", "Ticket", "Cabin"],
        )
        self.X, self.y, self.feature_names = self.data_loader.load()
        self.Cs = np.logspace(-4, 4, 10)

    def test_classification_lasso_path(self):
        np.random.seed(0)
        coefs = classification_lasso_path(self.X, self.y, self.Cs)
        n_zeros = np.sum(np.abs(coefs) < 1e-3, axis=1).ravel()

        corr, _ = stats.spearmanr(n_zeros, self.Cs)
        assert corr < -0.5


if __name__ == "__main__":
    unittest.main()
