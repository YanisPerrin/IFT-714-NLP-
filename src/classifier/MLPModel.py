from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir) 

from classifier.Classifier import Classifier

class MLPModel(Classifier):
    """
    MLP (Multi-Layer Perceptron) classifier with grid search and validation curve parameters.
    """

    def __init__(self):
        """
        Initializes an MLPModel with default parameters.
        """
        super().__init__("MLP", 
                         MLPClassifier(max_iter=1500, random_state=42),
                         {
                             'alpha': np.logspace(-4, 0, 5),
                             'hidden_layer_sizes': [(99,), (99, 99), (99, 99, 99)]
                         },
                         {
                             'alpha': np.logspace(-4, 2, 5),
                         }
                )