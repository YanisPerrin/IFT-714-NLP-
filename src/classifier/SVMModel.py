from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir) 

from classifier.Classifier import Classifier

class SVMModel(Classifier):
    """
    Support Vector Machine (SVM) classifier with grid search and validation curve parameters.
    """

    def __init__(self):
        """
        Initializes an SVMModel with default parameters.
        """
        super().__init__("SVM", 
                         SVC(random_state=42),
                         {
                             'C': np.logspace(-3, 3, 7),
                             'gamma': np.logspace(-3, 1, 5),
                             'kernel': ['rbf', 'poly']
                         },
                         {
                             'gamma': np.logspace(-3, 1, 5),
                         }
                )