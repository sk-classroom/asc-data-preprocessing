import unittest
import numpy as np
import sys

sys.path.append("assignments/")
from assignment import *
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import os


class TestLogisticRegression(unittest.TestCase):
    """
    Unit test for the LogisticRegression class.
    """

    def setUp(self):
        """
        Set up the test environment for each unit test for LogisticRegression.
        """
        self.n_features = 5
        self.logistic_regression = LogisticRegression(
            eta=0.01, n_iter=50, random_state=1
        )
        self.X, self.y = make_blobs(
            n_samples=100, centers=2, n_features=self.n_features, random_state=0
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def test_fit(self):
        """
        Test the fit method of LogisticRegression.
        """
        self.logistic_regression.fit(self.X_train, self.y_train)
        self.assertIsNotNone(self.logistic_regression.w_)
        self.assertIsNotNone(self.logistic_regression.b_)
        self.assertEqual(len(self.logistic_regression.w_), self.n_features)

    def test_predict(self):
        """
        Test the predict method of LogisticRegression.
        """
        self.logistic_regression.fit(self.X_train, self.y_train)
        predictions = self.logistic_regression.predict(self.X_test)
        self.assertTrue(np.sum(predictions != self.y_test) <= len(self.y_test) * 0.2)


class TestLogisticRegressionRidge(unittest.TestCase):
    """
    Unit test for the LogisticRegressionRidge class.
    """

    def setUp(self):
        """
        Set up the test environment for each unit test for LogisticRegressionRidge.
        """
        self.n_features = 5
        self.logistic_regression = LogisticRegressionRidge(
            eta=0.01, n_iter=50, gamma=0.1, random_state=1
        )
        self.X, self.y = make_blobs(
            n_samples=100, centers=2, n_features=self.n_features, random_state=0
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def test_fit(self):
        """
        Test the fit method of LogisticRegressionRidge.
        """
        self.logistic_regression.fit(self.X_train, self.y_train)
        self.assertIsNotNone(self.logistic_regression.w_)
        self.assertIsNotNone(self.logistic_regression.b_)
        self.assertEqual(len(self.logistic_regression.w_), self.n_features)

    def test_predict(self):
        """
        Test the predict method of LogisticRegressionRidge.
        """
        self.logistic_regression.fit(self.X_train, self.y_train)
        predictions = self.logistic_regression.predict(self.X_test)
        self.assertTrue(np.sum(predictions != self.y_test) <= len(self.y_test) * 0.2)

    def test_regularization(self):
        """
        Test the effect of regularization in LogisticRegressionRidge.
        """
        logistic_regression_small_gamma = LogisticRegressionRidge(
            eta=0.01, n_iter=50, gamma=0.01, random_state=1
        )
        logistic_regression_small_gamma.fit(self.X_train, self.y_train)

        logistic_regression_large_gamma = LogisticRegressionRidge(
            eta=0.01, n_iter=50, gamma=1, random_state=1
        )
        logistic_regression_large_gamma.fit(self.X_train, self.y_train)

        norm_small_gamma = np.linalg.norm(logistic_regression_small_gamma.w_)
        norm_large_gamma = np.linalg.norm(logistic_regression_large_gamma.w_)

        self.assertTrue(norm_small_gamma > norm_large_gamma)


class TestOneVsRest(unittest.TestCase):
    """
    Unit test for the OneVsRest class.
    """

    def setUp(self):
        """
        Set up the test environment for each unit test for OneVsRest.
        """
        self.n_classes = 3
        self.logistic_regression = LogisticRegression(
            eta=0.01, n_iter=50, random_state=1
        )
        self.one_vs_rest = OneVsRest(self.logistic_regression, self.n_classes)
        self.X, self.y = make_blobs(
            n_samples=100, centers=self.n_classes, random_state=0
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def test_fit(self):
        """
        Test the fit method of OneVsRest.
        """
        self.one_vs_rest.fit(self.X_train, self.y_train)
        for classifier in self.one_vs_rest.classifiers:
            self.assertIsNotNone(classifier.w_)
            self.assertIsNotNone(classifier.b_)

    def test_predict(self):
        """
        Test the predict method of OneVsRest.
        """
        self.one_vs_rest.fit(self.X_train, self.y_train)
        predictions = self.one_vs_rest.predict(self.X_test)
        self.assertTrue(np.sum(predictions != self.y_test) <= len(self.y_test) * 0.2)


if __name__ == "__main__":
    """
    Main entry point for the script.
    """
    unittest.main()
