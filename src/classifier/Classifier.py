from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

class Classifier:
    """
    A class for building and evaluating machine learning classifiers.
    """

    def __init__(self, name, model, gs_parameters, validation_parameters):
        """
        Initializes a Classifier with a name, model, grid search parameters, and validation parameters.
        """
        self.name = name
        self.model = model
        self.gs_parameters = gs_parameters
        self.validation_parameters = validation_parameters

    def train(self, X, y):
        """
        Trains the classifier on the given training data.
        """
        self.model.fit(X, y)

    def test(self, X, y, scoring="accuracy"):
        """
        Tests the classifier on the given test data using the specified scoring metric.
        """
        if scoring == "roc_auc_ovr_weighted":
            predictions = self.model.predict_proba(X)
            score = roc_auc_score(y, predictions, multi_class='ovr')
        elif scoring == "f1_score":
            predictions = self.model.predict(X)
            score = f1_score(y, predictions, average="weighted")
        else:
            predictions = self.model.predict(X)
            score = accuracy_score(y, predictions)
        return score

    def gridsearch(self, X, y, scoring):
        """
        Performs grid search using GridSearchCV to find the best hyperparameters for the classifier.
        """
        cv = KFold(n_splits=3)
        grid_search = GridSearchCV(self.model, self.gs_parameters, cv=cv, scoring=scoring, verbose=1, n_jobs=-1)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        print("Best Grid Search parameters for " + self.name + " model: ", best_params)
        self.model.set_params(**best_params)

    def get_model(self):
        """
        Returns the trained model.
        """
        return self.model

    def set_model(self, new_model):
        """
        Sets the model to a new model.
        """
        self.model = new_model

    def get_validation_parameters(self):
        """
        Returns the parameters for plotting validation curves.
        """
        return self.validation_parameters