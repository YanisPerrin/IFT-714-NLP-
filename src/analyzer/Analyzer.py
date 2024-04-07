import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ValidationCurveDisplay, LearningCurveDisplay

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir) 



class Analyzer():
    """
    A class for analyzing datasets and visualizing model performance.
    """

    def statistics(self, dataset):
        """
        Returns the statistics of the dataset using the describe function.
        """
        return dataset.describe()
    
    def get_shape(self, dataset):
        """
        Returns the shapes of the dataset using the shape function.
        """
        return dataset.shape
    
    def number_of_columns_by_type(self, dataset):
        """
        Returns the number of columns according to their type.
        """
        return dataset.dtypes.value_counts()
    
    def number_na(self, dataset):
        """
        Returns the number of missing values in the dataset.
        """
        return dataset.isna().sum().sum()

    def number_duplicated(self, dataset):
        """
        Returns the number of duplicated rows in the dataset.
        """
        return dataset.duplicated().sum()
    
    def number_of_observations_per_class(self, dataset, column):
        """
        Plots the bar plot of the dataset based on the number of observations per class.
        """
        fig, axes = plt.subplots(5, 2, figsize=(10, 20))
        for i, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                start_idx = i * 2 + j 
                end_idx = start_idx + 1  
                count_by_class = dataset[column].value_counts()[start_idx * 10:end_idx * 10]
                count_by_class.plot(kind="bar", ax=ax)
        plt.tight_layout()
        plt.show()

    def percentage_na_by_column(self, dataset):
        """
        Returns the percentage of NaN values in each column of the dataset.
        """
        return (dataset.isna().sum() * 100) / len(dataset)

    def counter_values(self, dataset, column):
        """
        Returns the number of occurrences for each unique value in the specified column.
        """
        return Counter(dataset[column])

    def histogram(self, dataset, column):
        """
        Plots the histogram of the dataset based on the specified column.
        """
        fig = px.histogram(dataset, x=column)
        fig.show()

    def boxplot(self, dataset, column):
        """
        Plots the boxplot of the dataset based on the specified column.
        """
        fig = px.box(dataset, y=column)
        fig.show()

    def plot_validation_curves(self, models, X, y, param_name, param_range, cv=KFold(n_splits=5)):
        """
        Plots the validation curves of the specified models based on the targeted parameters.
        """
        from data_manager.Preprocessor import Preprocessor
        X = Preprocessor.standardization(Preprocessor(), X, X.columns)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, model in enumerate(models):
            display = ValidationCurveDisplay.from_estimator(
                model,
                X,
                y,
                param_name=param_name[i],
                param_range=param_range[i],
                cv=cv,
                scoring="accuracy",
                ax=axes[i],
                n_jobs=-1
            )
            axes[i].set_title(f'Validation Curve ({model.__class__.__name__})')
            handles, label = axes[i].get_legend_handles_labels()
            axes[i].legend(handles[:2], ["Training Score", "Validation Score"])
        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self, models, X, y, cv=KFold(n_splits=5)):
        """
        Plots the learning curves of the specified models.
        """
        from data_manager.Preprocessor import Preprocessor
        X = Preprocessor.standardization(Preprocessor(), X, X.columns)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, model in enumerate(models):
            display = LearningCurveDisplay.from_estimator(
                model, X, y, cv=cv, scoring="accuracy", ax=axes[i], n_jobs=-1
            )
            axes[i].set_title(f'Learning Curve ({model.__class__.__name__})')
            handles, label = axes[i].get_legend_handles_labels()
            axes[i].legend(handles[:2], ["Training Score", "Validation Score"])
        plt.tight_layout()
        plt.show()

    def plot_roc_curve_multilabel(self, models, X, y):
        """
        Plots the ROC curve for each class of the target variable.
        """
        n_classes = len(np.unique(y))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, model in enumerate(models):
            if model.__class__.__name__ in ["LogisticRegression", "SVC"]:
                y_score = model.decision_function(X)
            else:
                y_score = model.predict_proba(X)
            y_binarize = label_binarize(y, classes=np.unique(y))
            for j in range(n_classes):
                fpr, tpr, _ = roc_curve(y_binarize[:, j], y_score[:, j])
                roc_auc = auc(fpr, tpr)
                axes[i].plot([0, 1], [0, 1], 'k--')
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(f'ROC curve ({model.__class__.__name__})')
                axes[i].plot(fpr, tpr, label=f'area = {roc_auc:.2f} for label {j}')
                axes[i].legend(loc='upper right', borderpad=0.1, bbox_to_anchor=(1.5, 1))
                axes[i].legend().set_visible(False)
        plt.show()

    def plot_mean_roc_curve(self, models, X, y):
        """
        Plots the mean ROC curve for each model.
        """
        n_classes = len(np.unique(y))
        plt.figure(figsize=(12, 8))
        for model in models:
            if model.__class__.__name__ in ["LogisticRegression", "SVC"]:
                y_score = model.decision_function(X)
            else:
                y_score = model.predict_proba(X)
            y_binarize = label_binarize(y, classes=np.unique(y))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_binarize[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.mean([np.interp(mean_fpr, fpr[i], tpr[i]) for i in range(n_classes)], axis=0)
            mean_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, linestyle='--', label=f'Mean ROC curve ({model.__class__.__name__}) (area = {mean_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Mean ROC curve for each model')
        plt.legend(loc='lower right')
        plt.show()

    def plot_scores(self, scores, models):
        """
        Plots the different scores of each model for each cross-validation fold.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        x = ['1st fold', '2nd fold', '3rd fold', '4th fold', '5th fold']
        titles = ["Train Accuracy", "Validation Accuracy", "Test Accuracy", "F1_score"]
        for i, score in enumerate(scores):
            for j in range(len(score)):
                axes[i].plot(x, score[j], 'o-', label=models[j].__class__.__name__)
                axes[i].set_title(titles[i])
                axes[i].set_ylabel("Score")
                axes[i].set_ylim([0, 1.05])
                axes[i].legend(loc="best")
        plt.tight_layout()
        plt.show()

    def barplot_mean_score(self, data, columns, target_column):
        """
        Plots the barplot of the mean score of the different cross-validation folds for each model.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        for i, column in enumerate(columns):
            axes[i].set_title(column)
            axes[i].set_xlabel("Score")
            sns.barplot(y=target_column, x=column, data=data, ax=axes[i])
        plt.tight_layout()
        plt.show()
        
    def plot_pca_variance(self, pca):
        """
        Plots the barplot of the ratio of the variance explained by the different principal components.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
        plt.ylabel('Explained variance')
        plt.xlabel('Principal components')
        plt.title('Variance explained by the principal components')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
        plt.ylabel('Cumulative variance explained')
        plt.xlabel('Number of principal component')
        plt.title('Cumulative variance explained by the principal components')
        plt.show()
        
    def plot_pca_visualtion_2cp(self, df_pca, target):
        """
        Plots the points of the two first principal components.
        """
        fig, ax = plt.subplots()
        sns.scatterplot(ax=ax,x=df_pca['PC1'], y=df_pca['PC2'], hue=df_pca[target.name], palette='viridis')
        ax.set_title("ACP - Visualization of all classes on the first two principal components")
        ax.set_xlabel('First principal component (PC1)')
        ax.set_ylabel('Second principal component (PC2)')
        ax.legend(target.unique())
        plt.show()
        
    def plot_heatmap_correlation(self, corr_matrix):
        """
        Plots the heatmap of the correlation matrix.
        """
        fig, axes = plt.subplots(5, 2, figsize=(10, 20))
        for i, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                start_idx = i * 2 + j 
                end_idx = start_idx + 1  
                part_of_corr_matrix = corr_matrix.iloc[start_idx * 10:end_idx * 10, start_idx * 10:end_idx * 10]
                sns.heatmap(part_of_corr_matrix, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        plt.title('Matrice de corrélation des meilleures caractéristiques - Global')
        plt.tight_layout()
        plt.show()