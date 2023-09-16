import numpy as np  # NumPy for numerical operations
from collections import Counter  # Counter for counting votes in the majority voting process

def euclidean_distance(x1, x2):
    """
    Calculate the Euclidean distance between two data points x1 and x2.

    Parameters:
    x1 (numpy.ndarray): The first data point.
    x2 (numpy.ndarray): The second data point.

    Returns:
    float: The Euclidean distance between x1 and x2.
    """
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, k=3):
        """
        Initialize the KNN classifier.

        Parameters:
        k (int): The number of neighbors to consider (default is 3).
        """
        self.k = k

    def fit(self, X, y):
        """
        Fit the KNN classifier with training data.

        Parameters:
        X (numpy.ndarray): The training data features.
        y (numpy.ndarray): The training data labels.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict labels for a set of data points X.

        Parameters:
        X (numpy.ndarray): The data points for which to make predictions.

        Returns:
        list: A list of predicted labels.
        """
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        """
        Predict the label for a single data point x.

        Parameters:
        x (numpy.ndarray): The data point for which to make a prediction.

        Returns:
        int: The predicted label for the input data point x.
        """
        # Compute the distances to all training data points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the indices of the k-nearest training data points
        k_indices = np.argsort(distances)[:self.k]

        # Get the labels of the k-nearest training data points
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Perform majority vote to determine the final label
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
