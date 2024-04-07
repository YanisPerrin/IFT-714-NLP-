from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir) 

from classifier.Classifier import Classifier

class RandomForestModel(Classifier):
    """
    Random Forest classifier with grid search and validation curve parameters.
    """

    def __init__(self):
        """
        Initializes a RandomForestModel with default parameters.
        """
        super().__init__("Random Forest", 
                         RandomForestClassifier(random_state=42),
                         {
                             'n_estimators': [10, 50, 100, 500],
                             'criterion': ['gini', 'entropy'],
                             'max_depth': [3, 5, 8, 10, 15]
                         },
                         {
                             'max_depth': [3, 5, 8, 10, 15]
                         }
                )