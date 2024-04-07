from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir) 

from classifier.Classifier import Classifier

class LogisticRegressionModel(Classifier):
    """
    Logistic Regression classifier with grid search and validation curve parameters.
    """

    def __init__(self):
        """
        Initializes a LogisticRegressionModel with default parameters.
        """
        super().__init__("Logistic Regression", 
                         LogisticRegression(max_iter=1000, penalty='l2', random_state=42),
                         {
                             'solver': ['lbfgs', 'liblinear'], 
                             'C': np.logspace(-3, 3, 7)
                         },
                         {
                             'C': np.logspace(-3, 3, 7)
                         }
                )