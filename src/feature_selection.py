import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

class RecursiveFeatureEliminator(BaseEstimator, TransformerMixin):
    '''
    A transformer class for selecting features based on the Recursive Feature Elimination (RFE) technique.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by recursively selecting the features with highest feature 
        importances until a final desired number of features is obtained through stratified k-fold cross validation.
    '''

    def __init__(self, estimator=LGBMClassifier(), scoring='roc_auc', n_folds=5):
        '''
        Initialize the Recursive Feature Elimination (RFE) transformer.
        
        Args:
            estimator (object, default=LGBMCLassifier): The model to obtain feature importances.
            n_folds (int, default=5): The number of folds for stratified k-fold cross validation.
            scoring (object, default='roc_auc'): The scoring for cross-validation.
            
        '''
        stratified_kfold = StratifiedKFold(n_splits=n_folds, 
                                           shuffle=True, 
                                           random_state=42)
        self.rfe = RFECV(estimator=estimator, 
                         cv=stratified_kfold,
                         scoring=scoring)

    def fit(self, X, y):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like): Target labels.

        Returns:
            self: Returns an instance of self.
        '''
        self.rfe.fit(X, y)
        
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by recursively selecting the features with highest feature 
        importances.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after recursively selecting the features with highest feature 
            importances.
        '''
        # Recursively select the features with highest feature importances.
        X_selected = self.rfe.transform(X)

        # Create a dataframe for the final selected features.
        selected_df = pd.DataFrame(X_selected,
                                  columns=self.rfe.get_feature_names_out())

        return selected_df
