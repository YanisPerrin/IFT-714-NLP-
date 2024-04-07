from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir) 

from classifier.Classifier import Classifier

class KNNModel(Classifier):
    """
    K Nearest Neighbors classifier with grid search and validation curve parameters.
    """

    def __init__(self):
        """
        Initializes a KNNModel with default parameters.
        """
        super().__init__("K Nearest Neighbors", 
                         KNeighborsClassifier(),
                         {
                             'n_neighbors': [1, 2, 5, 8, 10],
                             'weights': ['uniform', 'distance']
                         },
                         {
                             'n_neighbors': [1, 2, 5, 8, 10],
                         }
                )